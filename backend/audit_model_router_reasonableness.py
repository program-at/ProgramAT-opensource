"""Audit whether model_router choices look reasonable for existing tools.

This script statically scans tools/*.py for prior direct model usage and
router-client calls, then asks backend.model_router which profile it would pick
for the same task. It does not call any external model APIs.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
TOOLS_DIR = REPO_ROOT / "tools"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import model_router  # noqa: E402


EXPECTED_BY_TASK = {
    "image_analysis": "gpt4o",
    "visual_understanding": "gpt4o",
    "visual_reasoning": "gpt4o",
    "ocr": "google_vision_ocr",
    "object_detection": "yolo11_detector",
    "object_localization": "yolo11_detector",
    "simple_parsing": "gemini_flash_lite",
    "text_parse": "gemini_flash_lite",
    "tool_retrieval": "gemini_flash_lite",
    "summarization": "gemini_flash_lite",
    "code_generation": "llama",
    "code_repair": "llama",
    "general_reasoning": "gemini_flash",
}


@dataclass
class AuditCase:
    tool: str
    source: str
    line: int
    reason: str
    capability: str
    metadata: dict[str, Any]
    expected_profile: str


@dataclass
class AuditResult:
    tool: str
    source: str
    line: int
    reason: str
    capability: str
    route_text: str | None
    labels: list[str]
    expected_profile: str
    selected_profile: str
    selected_model: str
    selected_backend: str
    provider: str
    status: str
    detail: str


def _literal(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.List):
        return [_literal(item, constants) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(item, constants) for item in node.elts)
    if isinstance(node, ast.Dict):
        parsed: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = _literal(key_node, constants) if key_node is not None else None
            if isinstance(key, str):
                parsed[key] = _literal(value_node, constants)
        return parsed
    return None


def _module_constants(tree: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _literal(node.value, constants)
            if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
                constants[node.targets[0].id] = value
    return constants


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _as_label_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return []


def _expected_detector_profile(model_name: str, labels: list[str]) -> str:
    lowered = model_name.lower()
    if "world" in lowered:
        return "yolo_world_detector"
    if labels and any(label not in model_router.COCO_CLASSES for label in labels):
        return "yolo_world_detector"
    return "yolo11_detector"


def _detector_case(path: Path, call: ast.Call, constants: dict[str, Any]) -> AuditCase | None:
    if _call_name(call) != "YOLO":
        return None
    if not call.args:
        return None

    model_name = _literal(call.args[0], constants)
    if not isinstance(model_name, str):
        return None

    labels: list[str] = []
    route_text = "pure object_detection for common objects"
    capability = "object_detection"
    if "world" in model_name.lower():
        labels = _as_label_list(constants.get("DOOR_CLASSES") or constants.get("TARGET_CLASSES"))
        capability = "object_localization"
        label_text = ", ".join(labels[:8]) if labels else "open vocabulary objects"
        route_text = f"pure object_localization for labels {label_text}"

    metadata = {
        "tool_name": path.stem,
        "labels": labels,
        "route_text": route_text,
    }
    return AuditCase(
        tool=path.stem,
        source=str(path.relative_to(REPO_ROOT)),
        line=call.lineno,
        reason=f"prior direct detector usage: YOLO({model_name!r})",
        capability=capability,
        metadata=metadata,
        expected_profile=_expected_detector_profile(model_name, labels),
    )


def _router_call_case(path: Path, call: ast.Call, constants: dict[str, Any]) -> AuditCase | None:
    call_name = _call_name(call)
    if call_name not in {"llm_call", "vision_call", "routed_llm_call", "routed_vision_call"}:
        return None

    capability_node = (
        _keyword(call, "capability")
        or _keyword(call, "task")
        or _keyword(call, "task_category")
    )
    capability = _literal(capability_node, constants) if capability_node is not None else None
    if not isinstance(capability, str) or not capability:
        return None

    metadata = {}
    metadata_node = _keyword(call, "metadata")
    parsed_metadata = _literal(metadata_node, constants) if metadata_node is not None else None
    if isinstance(parsed_metadata, dict):
        metadata.update(parsed_metadata)
    metadata.setdefault("tool_name", path.stem)

    if call_name in {"vision_call", "routed_vision_call"}:
        prompt = _literal(_keyword(call, "prompt"), constants)
        if isinstance(prompt, str) and prompt.strip():
            metadata.setdefault("route_text", prompt)

    expected_profile = EXPECTED_BY_TASK.get(capability)
    if expected_profile is None:
        return None

    return AuditCase(
        tool=path.stem,
        source=str(path.relative_to(REPO_ROOT)),
        line=call.lineno,
        reason=f"router-client {call_name} task",
        capability=capability,
        metadata=metadata,
        expected_profile=expected_profile,
    )


def _ocr_case(path: Path, call: ast.Call) -> AuditCase | None:
    if _call_name(call) != "detect_text_google_vision":
        return None
    return AuditCase(
        tool=path.stem,
        source=str(path.relative_to(REPO_ROOT)),
        line=call.lineno,
        reason="prior direct OCR usage: detect_text_google_vision",
        capability="ocr",
        metadata={
            "tool_name": path.stem,
            "route_text": "extract readable text from image using OCR",
        },
        expected_profile="google_vision_ocr",
    )


def discover_cases(tools_dir: Path = TOOLS_DIR) -> list[AuditCase]:
    cases: list[AuditCase] = []
    for path in sorted(tools_dir.glob("*.py")):
        if path.name in {"model_router_client.py", "litellm_utils.py"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print(f"Skipping {path}: syntax error: {exc}", file=sys.stderr)
            continue

        constants = _module_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for builder in (
                lambda p, c, const: _detector_case(p, c, const),
                lambda p, c, const: _router_call_case(p, c, const),
                lambda p, c, const: _ocr_case(p, c),
            ):
                case = builder(path, node, constants)
                if case is not None:
                    cases.append(case)

    return cases


def run_case(case: AuditCase) -> AuditResult:
    route_info = model_router.get_route_info(case.capability, dict(case.metadata))
    profile_data = route_info.get("profile_data") or {}
    selected_backend = model_router._profile_backend(profile_data)
    selected_profile = route_info["selected_profile"]
    status = "pass" if selected_profile == case.expected_profile else "fail"
    detail = "selected expected profile"
    if status == "fail":
        detail = f"expected {case.expected_profile}, got {selected_profile}"

    return AuditResult(
        tool=case.tool,
        source=case.source,
        line=case.line,
        reason=case.reason,
        capability=route_info["task_category"],
        route_text=case.metadata.get("route_text"),
        labels=_as_label_list(case.metadata.get("labels")),
        expected_profile=case.expected_profile,
        selected_profile=selected_profile,
        selected_model=route_info["selected_model"],
        selected_backend=selected_backend,
        provider=route_info["provider"],
        status=status,
        detail=detail,
    )


def _print_report(results: list[AuditResult]) -> None:
    print("Model Router Reasonableness Audit")
    print("=" * 35)
    for result in results:
        marker = "PASS" if result.status == "pass" else "FAIL"
        location = f"{result.source}:{result.line}"
        print(f"{marker} {result.tool} ({location})")
        print(f"  reason: {result.reason}")
        print(f"  task: {result.capability}")
        if result.route_text:
            print(f"  route_text: {result.route_text}")
        if result.labels:
            print(f"  labels: {', '.join(result.labels[:10])}")
        print(
            "  selected: "
            f"{result.selected_profile} / {result.selected_backend} / "
            f"{result.selected_model} ({result.provider})"
        )
        print(f"  expected: {result.expected_profile}")
        print()

    failures = [result for result in results if result.status != "pass"]
    print(f"Summary: {len(results) - len(failures)} passed, {len(failures)} failed, {len(results)} checked")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", type=Path, default=TOOLS_DIR, help="Directory containing tool Python files.")
    parser.add_argument("--json-out", type=Path, help="Optional path for machine-readable audit output.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    args = parser.parse_args()

    original_mode = model_router.ROUTING_MODE
    model_router.ROUTING_MODE = "semantic"
    try:
        cases = discover_cases(args.tools_dir)
        results = [run_case(case) for case in cases]
    finally:
        model_router.ROUTING_MODE = original_mode

    failures = [result for result in results if result.status != "pass"]

    if args.json_out:
        payload = {
            "summary": {
                "checked": len(results),
                "passed": len(results) - len(failures),
                "failed": len(failures),
            },
            "results": [asdict(result) for result in results],
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.quiet:
        print(f"model-router audit: {len(results) - len(failures)} passed, {len(failures)} failed, {len(results)} checked")
    else:
        _print_report(results)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
