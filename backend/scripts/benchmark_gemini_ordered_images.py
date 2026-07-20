#!/usr/bin/env python3
"""Compare 4/6/8 uniformly sampled JPEGs across saved card scenarios."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from nvidia_hosted_client import (  # noqa: E402
    NvidiaHostedError, parse_structured_video_result, probe_mp4,
    validate_played_card_event,
)
from scripts.benchmark_gemini_video_inputs import generation_config  # noqa: E402
from scripts.compare_last_hosted_clip import normalize_expected_card, serializable  # noqa: E402
from scripts.test_nvidia_hosted_video import played_card_prompt  # noqa: E402

MODEL = "gemini-3.1-flash-lite-preview"
RUNS = 5
FRAME_COUNTS = (4, 6, 8)
REPORT_PATH = BACKEND_DIR / "debug" / "gemini_ordered_image_frame_count_benchmark.json"

# Human-audited from ordered contact sheets. Do not add inferred/model-labeled scenarios.
SCENARIOS = {
    "clear_play": {
        "path": BACKEND_DIR / "hosted_video_debug/played_card-7a21259f-clip-0007/clip.mp4",
        "expected_event": True, "expected_card": "nine of hearts",
    },
    "no_card_played": {
        "path": BACKEND_DIR / "hosted_video_debug/played_card-7a21259f-clip-0006/clip.mp4",
        "expected_event": False, "expected_card": None,
    },
    "fast_play": {
        "path": BACKEND_DIR / "hosted_video_debug/played_card-dbb0e9a1-clip-0003/clip.mp4",
        "expected_event": True, "expected_card": "jack of diamonds",
    },
    "partial_occlusion": {
        "path": BACKEND_DIR / "hosted_video_debug/played_card-dbb0e9a1-clip-0004/clip.mp4",
        "expected_event": True, "expected_card": "nine of hearts",
    },
}

RANKS = ("ace", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "jack", "queen", "king")
SUITS = ("clubs", "diamonds", "hearts", "spades")
CARDS = tuple(f"{rank} of {suit}" for rank in RANKS for suit in SUITS)


def ffmpeg() -> str:
    executable = shutil.which(os.getenv("PROGRAMAT_FFMPEG_BINARY", "ffmpeg"))
    if not executable:
        raise RuntimeError("FFmpeg is required")
    return executable


def sample_uniform_jpegs(source: Path, directory: Path, count: int) -> tuple[list[Path], float]:
    """Sample exact ordered positions from t=0 through the full clip endpoint."""
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    metadata = probe_mp4(source, os.getenv("PROGRAMAT_FFMPEG_BINARY", "ffmpeg"))
    stream = (metadata.get("streams") or [{}])[0]
    frame_total = int(stream.get("nb_frames") or 0)
    if frame_total < count:
        raise RuntimeError(f"Clip has {frame_total} frames; cannot sample {count}")
    indices = [round(index * (frame_total - 1) / (count - 1)) for index in range(count)]
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    subprocess.run([
        ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", f"select='{selection}',scale=720:1280:force_original_aspect_ratio=decrease",
        "-vsync", "0", "-q:v", "3", "-y", str(directory / "frame-%02d.jpg"),
    ], check=True)
    paths = sorted(directory.glob("frame-*.jpg"))
    if len(paths) != count:
        raise RuntimeError(f"Expected {count} sampled frames, produced {len(paths)}")
    return paths, (time.perf_counter() - started) * 1000


def explicitly_wrong_removed_card(evidence: str, played: str) -> str | None:
    text = " ".join(evidence.casefold().split())
    removal_words = r"(?:played|removed|disappeared|absent|taken|left the hand)"
    for card in CARDS:
        if card == played or card not in text:
            continue
        card_pattern = re.escape(card)
        if (re.search(rf"{card_pattern}\s+(?:(?:was|is|has been)\s+)?{removal_words}", text)
                or re.search(rf"{removal_words}\s+(?:the\s+)?{card_pattern}", text)):
            return card
    return None


def benchmark_validation(parsed: dict[str, Any]) -> tuple[bool, str]:
    _message, decision = validate_played_card_event(parsed, 0.8)
    if decision != "evidence_card_mismatch":
        return decision in {"accepted", "stable_no_event"}, decision
    played = normalize_expected_card(str(parsed.get("played_card") or ""))
    wrong = explicitly_wrong_removed_card(str(parsed.get("evidence") or ""), played)
    if wrong:
        return False, f"evidence_identifies_wrong_removed_card:{wrong}"
    # All other structural/card/confidence checks already passed before the
    # production validator reached evidence_card_mismatch.
    return True, "accepted_remaining_card_mentioned"


def score(raw: str, expected_event: bool, expected_card: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"raw_response": raw}
    try:
        parsed = parse_structured_video_result(raw)
        event = parsed.get("event_detected") if isinstance(parsed.get("event_detected"), bool) else None
        played = (
            normalize_expected_card(parsed["played_card"])
            if isinstance(parsed.get("played_card"), str) else None
        )
        validator_passed, validation = benchmark_validation(parsed)
        result.update({
            "parsed_json": parsed, "detected_event": event, "detected_card": played,
            "event_correct": event is expected_event,
            "played_card_correct": (
                played == normalize_expected_card(expected_card)
                if expected_event and expected_card else None
            ),
            "validator_passed": validator_passed, "validation": validation,
        })
    except NvidiaHostedError as exc:
        result.update({
            "parsed_json": None, "detected_event": None, "detected_card": None,
            "event_correct": False,
            "played_card_correct": False if expected_event else None,
            "validator_passed": False, "validation": "invalid_json",
            "parse_error": str(exc),
        })
    return result


async def run_variant(client, paths: list[Path], preprocessing_ms: float,
                      expected_event: bool, expected_card: str | None,
                      existing_runs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from google.genai import types
    parts = [types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg") for path in paths]
    prompt = played_card_prompt()
    config = generation_config()
    runs = []
    existing_by_index = {
        int(run.get("run", 0)): run for run in (existing_runs or [])
        if "api_error" not in run
    }
    for index in range(1, RUNS + 1):
        if index in existing_by_index:
            runs.append(existing_by_index[index])
            continue
        run = None
        for attempt in range(1, 9):
            request_started = time.perf_counter()
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=MODEL, contents=[prompt, *parts], config=config,
                )
                request_ms = (time.perf_counter() - request_started) * 1000
                run = score(response.text or "", expected_event, expected_card)
                usage = serializable(getattr(response, "usage_metadata", None))
                image_tokens = sum(
                    int(detail.get("token_count", 0))
                    for detail in (usage or {}).get("prompt_tokens_details", [])
                    if detail.get("modality") == "IMAGE"
                )
                run.update({
                    "run": index, "gemini_request_latency_ms": request_ms,
                    "end_to_end_latency_ms": preprocessing_ms + request_ms,
                    "usage": usage, "image_token_usage": image_tokens,
                })
                await asyncio.sleep(4.2)  # Stay below the 15 requests/minute free-tier limit.
                break
            except Exception as exc:
                if "429 RESOURCE_EXHAUSTED" in str(exc) and attempt < 8:
                    await asyncio.sleep(10)
                    continue
                run = {
                    "run": index, "gemini_request_latency_ms": None,
                    "end_to_end_latency_ms": None, "api_error": str(exc),
                    "event_correct": False,
                    "played_card_correct": False if expected_event else None,
                    "validator_passed": False,
                }
                break
        assert run is not None
        runs.append(run)
    return runs


def summarize(runs: list[dict[str, Any]], expected_event: bool) -> dict[str, Any]:
    completed = [run for run in runs if "api_error" not in run]
    def avg(key):
        values = [float(run[key]) for run in completed if run.get(key) is not None]
        return sum(values) / len(values) if values else None
    outcomes = [(run.get("detected_event"), run.get("detected_card")) for run in completed]
    return {
        "played_card_accuracy": (
            sum(run.get("played_card_correct") is True for run in runs) / len(runs)
            if expected_event else None
        ),
        "event_detection_accuracy": sum(run.get("event_correct") is True for run in runs) / len(runs),
        "strict_validator_pass_rate": sum(run.get("validator_passed") is True for run in runs) / len(runs),
        "consistent": bool(outcomes) and len(set(outcomes)) == 1,
        "outcomes": outcomes,
        "average_gemini_request_latency_ms": avg("gemini_request_latency_ms"),
        "average_end_to_end_latency_ms": avg("end_to_end_latency_ms"),
        "average_image_token_usage": avg("image_token_usage"),
    }


async def benchmark() -> int:
    load_dotenv(BACKEND_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    missing = [str(item["path"]) for item in SCENARIOS.values() if not item["path"].is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    from google import genai
    client = genai.Client(api_key=api_key)  # One client reused across all 60 calls.
    previous_report = {}
    if REPORT_PATH.is_file():
        try:
            previous_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_report = {}
    report: dict[str, Any] = {
        "model": MODEL, "runs_per_variant": RUNS,
        "frame_counts": list(FRAME_COUNTS), "resolution": [720, 1280],
        "uniform_full_span_sampling": True, "temperature": 0,
        "structured_json": True, "automatic_function_calling_disabled": True,
        "production_changed": False, "scenarios": {},
        "unavailable_scenarios": ["hand_leaving_frame"],
    }
    with tempfile.TemporaryDirectory(prefix="programat-ordered-images-") as directory:
        work = Path(directory)
        for scenario_name, scenario in SCENARIOS.items():
            frozen = work / scenario_name / "source.mp4"
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(scenario["path"], frozen)
            scenario_report = {
                "source_path": str(scenario["path"]),
                "source_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                "expected_event": scenario["expected_event"],
                "expected_card": scenario["expected_card"], "variants": {},
            }
            for count in FRAME_COUNTS:
                paths, preprocessing_ms = sample_uniform_jpegs(
                    frozen, work / scenario_name / f"{count}-frames", count
                )
                runs = await run_variant(
                    client, paths, preprocessing_ms,
                    scenario["expected_event"], scenario["expected_card"],
                    (((previous_report.get("scenarios") or {}).get(scenario_name) or {})
                     .get("variants", {}).get(str(count), {}).get("runs")),
                )
                scenario_report["variants"][str(count)] = {
                    "frame_count": count,
                    "input_bytes": sum(path.stat().st_size for path in paths),
                    "preprocessing_ms": preprocessing_ms,
                    "runs": runs,
                    "summary": summarize(runs, scenario["expected_event"]),
                }
            report["scenarios"][scenario_name] = scenario_report
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for scenario, data in report["scenarios"].items():
        print(f"\nscenario={scenario}")
        for count, variant in data["variants"].items():
            print(f"frames={count} bytes={variant['input_bytes']} preprocess_ms={variant['preprocessing_ms']:.1f} "
                  + json.dumps(variant["summary"], sort_keys=True))
    print(f"\nreport_path={REPORT_PATH}")
    return 0


def main() -> int:
    try:
        return asyncio.run(benchmark())
    except (FileNotFoundError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"benchmark_failed={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
