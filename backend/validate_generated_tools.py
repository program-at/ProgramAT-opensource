"""Static guardrails for generated ProgramAT tool files."""

from __future__ import annotations

import ast
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import yaml


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
    ("model file reference", re.compile(r"\.pt\b|['\"][^'\"]+\.pt['\"]")),
    ("model file discovery", re.compile(r"\b(?:glob|rglob)\s*\([^)]*\.pt[^)]*\)|os\.walk\s*\(")),
]

ALLOWED_SHARED_TOOL_FILES = {
    "litellm_utils.py",
    "model_router_client.py",
}

CAPABILITY_PROFILES_PATH = REPO_ROOT / "backend" / "capability_profiles.yaml"


def _load_canonical_task_categories() -> frozenset[str]:
    with CAPABILITY_PROFILES_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    capabilities = data.get("capabilities", {})
    if not isinstance(capabilities, dict) or not capabilities:
        raise ValueError(f"No capabilities configured in {CAPABILITY_PROFILES_PATH}")
    return frozenset(str(name) for name in capabilities)


CANONICAL_TASK_CATEGORIES = _load_canonical_task_categories()


def _extract_task_stages_section(issue_text: str) -> str:
    match = re.search(r"^##\s+Task\s+Stages\s*$", issue_text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""

    start = match.end()
    remainder = issue_text[start:]
    next_header = re.search(r"^##\s+", remainder, re.MULTILINE)
    if next_header:
        return remainder[: next_header.start()]
    return remainder


def extract_stage_capabilities(issue_text: str) -> List[str]:
    section = _extract_task_stages_section(issue_text)
    if not section:
        return []

    capabilities: List[str] = []
    for cap in re.findall(r"^\s*(?:[-*]\s*)?Capability\s*:\s*`?([a-z_]+)`?\s*$", section, re.IGNORECASE | re.MULTILINE):
        capability = cap.strip()
        if capability in CANONICAL_TASK_CATEGORIES:
            capabilities.append(capability)
    return capabilities


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


def validate_stage_enforcement(tool_text: str, issue_text: str, rel_path: Path) -> List[str]:
    failures: List[str] = []
    stage_capabilities = extract_stage_capabilities(issue_text)
    if not stage_capabilities:
        return failures

    actual_capabilities = extract_copilot_llm_task_categories(tool_text)
    if actual_capabilities != stage_capabilities:
        failures.append(
            f"{rel_path}: ordered copilot_llm_call capabilities {actual_capabilities} do not match "
            f"Task Stages {stage_capabilities}."
        )

    return failures


def validate_canonical_task_categories(tool_text: str, rel_path: Path) -> List[str]:
    failures: List[str] = []
    categories = extract_copilot_llm_task_categories(tool_text)
    unresolved_indexes = [index + 1 for index, category in enumerate(categories) if category is None]

    if unresolved_indexes:
        failures.append(
            f"{rel_path}: copilot_llm_call() missing resolvable capability at call(s) {unresolved_indexes}."
        )

    for index, category in enumerate(categories):
        if category is None:
            continue
        if category not in CANONICAL_TASK_CATEGORIES:
            failures.append(
                f"{rel_path}: non-canonical capability '{category}' in copilot_llm_call #{index + 1}."
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

        failures.extend(validate_canonical_task_categories(text, rel_path))
        if issue_text:
            failures.extend(validate_stage_enforcement(text, issue_text, rel_path))

    return failures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Specific tool files to validate.")
    parser.add_argument("--changed", metavar="BASE_REF", help="Validate changed tools relative to BASE_REF.")
    parser.add_argument("--all", action="store_true", help="Validate all Python tool files.")
    parser.add_argument("--issue-file", help="Optional issue markdown/body file used for Task Stages validation.")
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
        print("Generated tools must use model_router_client capability interfaces for LLM/VLM operations.")
        print("Do not implement detection, OCR, VLM, LLM, model loading, provider calls, or model discovery in tool files.")
        print(
            "When Task Stages are provided, compose one ordered copilot_llm_call() per stage."
        )
        print()
        for failure in failures:
            print(failure)
        return 1

    print("Generated tool router guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
