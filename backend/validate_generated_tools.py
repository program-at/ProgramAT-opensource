"""Static guardrails for generated ProgramAT tool files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


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


def validate_files(paths: Iterable[Path]) -> List[str]:
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

    return failures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Specific tool files to validate.")
    parser.add_argument("--changed", metavar="BASE_REF", help="Validate changed tools relative to BASE_REF.")
    parser.add_argument("--all", action="store_true", help="Validate all Python tool files.")
    args = parser.parse_args(argv)

    paths: List[Path] = []
    if args.all:
        paths.extend(TOOLS_DIR.glob("*.py"))
    if args.changed:
        paths.extend(_changed_tool_files(args.changed))
    if args.paths:
        paths.extend(_normalize_paths(args.paths))

    failures = validate_files(paths)
    if failures:
        print("Generated tools must use from model_router_client import llm_call for LLM/VLM operations.")
        print("Do not implement detection, OCR, VLM, LLM, model loading, provider calls, or model discovery in tool files.")
        print()
        for failure in failures:
            print(failure)
        return 1

    print("Generated tool router guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
