"""Static guardrails for generated ProgramAT tool files."""

from __future__ import annotations

import ast
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from policy_executor import ToolPolicyError, validate_tool_policy

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"

FORBIDDEN_PATTERNS = [
    ("import litellm", re.compile(r"^\s*(?:import|from)\s+litellm\b", re.MULTILINE)),
    ("litellm.completion", re.compile(r"\blitellm\s*\.\s*completion\s*\(")),
    ("direct ultralytics import", re.compile(r"^\s*(?:import\s+ultralytics\b|from\s+ultralytics\s+import\b)", re.MULTILINE)),
    ("direct YOLO import", re.compile(r"^\s*(?:import\s+YOLO\b|from\s+\S+\s+import\s+.*\bYOLO\b)", re.MULTILINE)),
    ("direct YOLO call", re.compile(r"\bYOLO\s*\(")),
    ("hardcoded YOLO model name", re.compile(r"\byolo11\w*\b", re.IGNORECASE)),
    ("direct Google Vision import", re.compile(r"^\s*(?:import\s+google\.cloud\.vision\b|from\s+google\.cloud\s+import\s+vision\b)", re.MULTILINE)),
    ("direct provider SDK import", re.compile(r"^\s*(?:import|from)\s+(?:openai|anthropic|google\.genai|google\.generativeai)\b", re.MULTILINE)),
    ("DEFAULT_MODEL constant", re.compile(r"\bDEFAULT_MODEL\b")),
    ("local ModelRouter class", re.compile(r"\bclass\s+ModelRouter\b")),
    ("COCO class list", re.compile(r"\bCOCO_CLASSES\b")),
    ("model registry", re.compile(r"\b(?:MODEL_REGISTRY|model_registry|available_models|provider_registry)\b")),
    ("provider fallback logic", re.compile(r"\b(?:fallback_models|fallback_model|provider_fallback|fallback_provider)\b")),
    ("custom model orchestration", re.compile(r"\b(?:execute_tool_policy|execute_resolved_tool_policy|call_model|ThreadPoolExecutor|as_completed)\b")),
    ("model file reference", re.compile(r"\.pt\b|['\"][^'\"]+\.pt['\"]")),
    ("model file discovery", re.compile(r"\b(?:glob|rglob)\s*\([^)]*\.pt[^)]*\)|os\.walk\s*\(")),
]

ALLOWED_SHARED_TOOL_FILES = {
    "litellm_utils.py",
    "tool_policy_client.py",
}

def _extract_string_constants(tree: ast.AST) -> dict[str, str]:
    constants: dict[str, str] = {}
    if not isinstance(tree, ast.Module):
        return constants

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                constants[target.id] = node.value.value
    return constants


def _extract_literal_constants(tree: ast.AST) -> dict[str, object]:
    constants: dict[str, object] = {}
    if not isinstance(tree, ast.Module):
        return constants
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return constants


def extract_copilot_llm_task_categories(tool_text: str) -> List[Optional[str]]:
    try:
        tree = ast.parse(tool_text)
    except SyntaxError:
        return []

    constants = _extract_string_constants(tree)
    categories: List[Optional[str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        is_copilot_call = False
        if isinstance(node.func, ast.Name) and node.func.id == "copilot_llm_call":
            is_copilot_call = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "copilot_llm_call":
            is_copilot_call = True

        if not is_copilot_call:
            continue

        category: Optional[str] = None
        for kw in node.keywords:
            if kw.arg not in {"capability", "task_category"}:
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                category = kw.value.value
            elif isinstance(kw.value, ast.Name):
                category = constants.get(kw.value.id)
            break

        categories.append(category)

    return categories


def validate_no_stringified_copilot_results(tool_text: str, rel_path: Path) -> List[str]:
    """Reject turning structured capability results into user-facing repr strings."""
    try:
        tree = ast.parse(tool_text)
    except SyntaxError:
        return []

    result_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        is_copilot_call = (
            isinstance(value.func, ast.Name) and value.func.id == "copilot_llm_call"
        ) or (
            isinstance(value.func, ast.Attribute) and value.func.attr == "copilot_llm_call"
        )
        if not is_copilot_call:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        result_names.update(target.id for target in targets if isinstance(target, ast.Name))

    failures = []
    for node in ast.walk(tree):
        stringified_name = None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"str", "repr"}
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in result_names
        ):
            stringified_name = node.args[0].id
        elif isinstance(node, ast.FormattedValue) and isinstance(node.value, ast.Name):
            if node.value.id in result_names:
                stringified_name = node.value.id

        if stringified_name:
            failures.append(
                f"{rel_path}:{node.lineno}: capability result '{stringified_name}' must not be "
                "stringified; return or use its ['response'] field for user-facing text."
            )
    return failures


def validate_take_photo_tool(tool_text: str, issue_text: str, rel_path: Path) -> List[str]:
    """Enforce the unified single-call take-photo generation contract."""
    try:
        tree = ast.parse(tool_text)
    except SyntaxError:
        return []
    constants = _extract_literal_constants(tree)
    if constants.get('EXECUTION_MODE') != 'take_photo':
        return []

    vlm_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "call_take_photo_vlm"
        )
    ]
    failures = []
    if len(vlm_calls) != 1:
        failures.append(
            f"{rel_path}: take-photo tools require exactly one call_take_photo_vlm() call; "
            f"found {len(vlm_calls)}."
        )
    tool_prompt = constants.get("TOOL_PROMPT")
    if tool_prompt is None:
        failures.append(f"{rel_path}: take-photo tools require one string TOOL_PROMPT constant.")
    if "TOOL_NAME" not in constants:
        failures.append(f"{rel_path}: take-photo tools require one string TOOL_NAME constant.")
    if constants.get('EXECUTION_MODE') != 'take_photo':
        failures.append(f"{rel_path}: static tools require EXECUTION_MODE = 'take_photo'.")
    if 'VIDEO_CONFIG' in constants:
        failures.append(f"{rel_path}: static tools must not declare temporal VIDEO_CONFIG.")
    prompt_uses = []
    for call in vlm_calls:
        prompt_keyword = next((kw for kw in call.keywords if kw.arg == "prompt"), None)
        prompt_uses.append(
            prompt_keyword is not None
            and isinstance(prompt_keyword.value, ast.Name)
            and prompt_keyword.value.id == "TOOL_PROMPT"
        )
    if prompt_uses != [True]:
        failures.append(f"{rel_path}: take-photo tools must pass TOOL_PROMPT directly as the helper prompt.")
    tool_name_uses = []
    for call in vlm_calls:
        keyword = next((kw for kw in call.keywords if kw.arg == "tool_name"), None)
        tool_name_uses.append(
            keyword is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "TOOL_NAME"
        )
    if tool_name_uses != [True]:
        failures.append(f"{rel_path}: take-photo tools must pass TOOL_NAME directly as tool_name.")
    if extract_copilot_llm_task_categories(tool_text):
        failures.append(f"{rel_path}: take-photo tools must not call copilot_llm_call().")
    if "TOOL_POLICY" in constants:
        try:
            validate_tool_policy(constants["TOOL_POLICY"])
        except (ToolPolicyError, TypeError) as exc:
            failures.append(f"{rel_path}: invalid TOOL_POLICY: {exc}.")
        policy_uses = []
        for call in vlm_calls:
            keyword = next((kw for kw in call.keywords if kw.arg == "policy"), None)
            policy_uses.append(
                keyword is not None and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "TOOL_POLICY"
            )
        if policy_uses != [True]:
            failures.append(
                f"{rel_path}: tools declaring TOOL_POLICY must pass policy=TOOL_POLICY "
                "to call_take_photo_vlm()."
            )
    elif any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "TOOL_POLICY" for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        ))
        for node in tree.body
    ):
        failures.append(f"{rel_path}: TOOL_POLICY must be static literal data.")
    return failures


def validate_rtvi_streaming_tool(
    tool_text: str, issue_text: str, rel_path: Path
) -> List[str]:
    """Require hosted-video tools to remain declarative and runtime-owned."""
    try:
        tree = ast.parse(tool_text)
    except SyntaxError:
        return []
    constants = _extract_literal_constants(tree)
    if constants.get('EXECUTION_MODE') != 'hosted_video_streaming':
        return []
    failures = []
    if not isinstance(constants.get('TOOL_NAME'), str):
        failures.append(f"{rel_path}: RTVI tools require one string TOOL_NAME.")
    if not isinstance(constants.get('TOOL_PROMPT'), str) or not constants.get('TOOL_PROMPT', '').strip():
        failures.append(f"{rel_path}: RTVI tools require one non-empty string TOOL_PROMPT.")
    video_config = constants.get('VIDEO_CONFIG')
    required_settings = {
        'window_seconds', 'interval_seconds', 'minimum_span_seconds',
        'minimum_unique_frames',
    }
    if not isinstance(video_config, dict):
        failures.append(f"{rel_path}: temporal tools require a literal VIDEO_CONFIG dictionary.")
    else:
        missing = sorted(required_settings - set(video_config))
        if missing:
            failures.append(f"{rel_path}: temporal VIDEO_CONFIG is missing: {', '.join(missing)}.")
        else:
            try:
                window = float(video_config['window_seconds'])
                interval = float(video_config['interval_seconds'])
                minimum_span = float(video_config['minimum_span_seconds'])
                minimum_frames = int(video_config['minimum_unique_frames'])
                if (window <= 0 or interval <= 0 or minimum_span <= 0
                        or minimum_span > window or minimum_frames < 2):
                    raise ValueError
            except (TypeError, ValueError):
                failures.append(f"{rel_path}: temporal VIDEO_CONFIG settings are invalid.")
    prompt = str(constants.get('TOOL_PROMPT') or '').casefold()
    temporal_terms = (
        'chronological', 'early', 'late', 'before', 'after', 'sequence',
        'duration', 'state change', 'changed', 'recent frames', 'video',
        'what just happened', 'history', 'multiple moments',
    )
    if prompt and not any(term in prompt for term in temporal_terms):
        failures.append(
            f"{rel_path}: temporal TOOL_PROMPT must explain chronological or state-change evidence."
        )
    output_config = constants.get('OUTPUT_CONFIG') or {}
    if isinstance(output_config, dict) and output_config.get('schema') == 'played_card_event':
        card_terms = ('played card', 'played_card', 'before_cards', 'after_cards')
        if not any(term in prompt for term in card_terms):
            failures.append(
                f"{rel_path}: played_card_event may only be used by a card-specific prompt."
            )
    forbidden = {
        'call_take_photo_vlm', 'copilot_llm_call', 'RtspPublisher',
        'NvidiaRtviClient', 'asyncio', 're', 'requests', 'aiohttp', 'ffmpeg',
    }
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    found = sorted(forbidden & used)
    imported = {
        alias.name.split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    found = sorted(set(found) | (forbidden & imported))
    if found:
        failures.append(
            f"{rel_path}: RTVI runtime owns streaming; forbidden tool symbols: "
            + ', '.join(found)
        )
    module_state = [
        target.id
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        and target.id not in {
            'TOOL_NAME', 'EXECUTION_MODE', 'TOOL_PROMPT', 'VIDEO_CONFIG', 'OUTPUT_CONFIG'
        }
    ]
    if module_state:
        failures.append(f"{rel_path}: RTVI tools must not keep module state: {module_state}")
    allowed_names = {
        'TOOL_NAME', 'EXECUTION_MODE', 'TOOL_PROMPT', 'VIDEO_CONFIG', 'OUTPUT_CONFIG'
    }
    non_declarative = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id in allowed_names
            for target in node.targets
        ):
            continue
        non_declarative.append(type(node).__name__)
    if non_declarative:
        failures.append(
            f"{rel_path}: RTVI tools may contain only declarative constants: "
            + ', '.join(non_declarative)
        )
    return failures


def _changed_tool_files(base_ref: str) -> List[Path]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base_ref}...HEAD", "--", "tools"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        raise

    return [REPO_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".py")]


def _normalize_paths(paths: Iterable[str]) -> List[Path]:
    result = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.suffix == ".py" and TOOLS_DIR in path.resolve().parents:
            result.append(path)
    return result


def validate_files(paths: Iterable[Path], issue_text: Optional[str] = None) -> List[str]:
    failures: List[str] = []
    for path in sorted(set(paths)):
        if path.name in ALLOWED_SHARED_TOOL_FILES or not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(REPO_ROOT)
        for label, pattern in FORBIDDEN_PATTERNS:
            match = pattern.search(text)
            if match:
                line_number = text.count("\n", 0, match.start()) + 1
                failures.append(f"{rel_path}:{line_number}: forbidden generated-tool pattern: {label}")

        failures.extend(validate_no_stringified_copilot_results(text, rel_path))
        if issue_text:
            failures.extend(validate_take_photo_tool(text, issue_text, rel_path))
            failures.extend(validate_rtvi_streaming_tool(text, issue_text, rel_path))

    return failures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Specific tool files to validate.")
    parser.add_argument("--changed", metavar="BASE_REF", help="Validate changed tools relative to BASE_REF.")
    parser.add_argument("--all", action="store_true", help="Validate all Python tool files.")
    parser.add_argument("--issue-file", help="Optional issue markdown/body used for mode validation.")
    args = parser.parse_args(argv)

    paths: List[Path] = []
    if args.all:
        paths.extend(TOOLS_DIR.glob("*.py"))
    if args.changed:
        paths.extend(_changed_tool_files(args.changed))
    if args.paths:
        paths.extend(_normalize_paths(args.paths))

    issue_text: Optional[str] = None
    if args.issue_file:
        issue_text = Path(args.issue_file).read_text(encoding="utf-8")

    failures = validate_files(paths, issue_text=issue_text)
    if failures:
        print("Generated tool validation failed.")
        print()
        for failure in failures:
            print(failure)
        return 1

    print("Generated tool guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
