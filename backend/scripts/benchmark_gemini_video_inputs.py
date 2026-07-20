#!/usr/bin/env python3
"""Benchmark three Gemini media representations of last_hosted_clip.mp4."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
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
    NvidiaHostedError,
    parse_structured_video_result,
    probe_mp4,
    validate_played_card_event,
)
from scripts.compare_last_hosted_clip import normalize_expected_card, serializable  # noqa: E402
from scripts.test_nvidia_hosted_video import played_card_prompt  # noqa: E402

CLIP_PATH = BACKEND_DIR / "debug" / "last_hosted_clip.mp4"
REPORT_PATH = BACKEND_DIR / "debug" / "gemini_input_latency_comparison.json"
MODEL = "gemini-3.1-flash-lite-preview"
RUNS = 3

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "event_detected", "before_cards", "after_cards", "played_card",
        "confidence", "evidence",
    ],
    "properties": {
        "event_detected": {"type": "boolean"},
        "before_cards": {"type": "array", "items": {"type": "string"}},
        "after_cards": {"type": "array", "items": {"type": "string"}},
        "played_card": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
}


def ffmpeg() -> str:
    executable = shutil.which(os.getenv("PROGRAMAT_FFMPEG_BINARY", "ffmpeg"))
    if not executable:
        raise RuntimeError("FFmpeg is required")
    return executable


def prepare_four_second_video(source: Path, destination: Path) -> float:
    started = time.perf_counter()
    metadata = probe_mp4(source, os.getenv("PROGRAMAT_FFMPEG_BINARY", "ffmpeg"))
    duration = float((metadata.get("format") or {}).get("duration") or 0)
    start = max(0.0, (duration - 4.0) / 2.0)
    subprocess.run([
        ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", str(start),
        "-i", str(source), "-t", "4", "-an", "-vf",
        "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-y", str(destination),
    ], check=True)
    return (time.perf_counter() - started) * 1000


def prepare_jpegs(source: Path, directory: Path, count: int = 8) -> tuple[list[Path], float]:
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    # fps=N/duration gives ordered, uniformly spaced source samples without an MP4 re-encode.
    metadata = probe_mp4(source, os.getenv("PROGRAMAT_FFMPEG_BINARY", "ffmpeg"))
    duration = float((metadata.get("format") or {}).get("duration") or 1)
    subprocess.run([
        ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", f"fps={count / duration},scale=720:1280:force_original_aspect_ratio=decrease",
        "-frames:v", str(count), "-q:v", "3", "-y", str(directory / "frame-%02d.jpg"),
    ], check=True)
    paths = sorted(directory.glob("frame-*.jpg"))
    if len(paths) != count:
        raise RuntimeError(f"Expected {count} JPEG frames, produced {len(paths)}")
    return paths, (time.perf_counter() - started) * 1000


def generation_config():
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=RESPONSE_SCHEMA,
        tools=[],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def classify(raw: str, expected: str) -> dict[str, Any]:
    result: dict[str, Any] = {"raw_response": raw}
    try:
        parsed = parse_structured_video_result(raw)
        _message, validation = validate_played_card_event(parsed, 0.8)
        if parsed.get("event_detected") is True and isinstance(parsed.get("played_card"), str):
            detected = normalize_expected_card(str(parsed["played_card"]))
        elif validation == "stable_no_event":
            detected = "no_event"
        else:
            detected = "rejected"
        result.update({
            "parsed_json": parsed,
            "validation": validation,
            "detected_result": detected,
            "card_correct": detected == expected,
            "validated_correct": validation in {"accepted", "stable_no_event"} and detected == expected,
        })
    except NvidiaHostedError as exc:
        result.update({
            "parsed_json": None, "validation": "invalid_json",
            "detected_result": "rejected", "card_correct": False,
            "validated_correct": False,
            "parse_error": str(exc),
        })
    return result


async def wait_for_file(client, uploaded):
    deadline = time.monotonic() + 120
    while getattr(getattr(uploaded, "state", None), "name", "") == "PROCESSING":
        if time.monotonic() >= deadline:
            raise TimeoutError("Gemini file processing timed out")
        await asyncio.sleep(1)
        uploaded = await asyncio.to_thread(client.files.get, name=uploaded.name)
    if getattr(getattr(uploaded, "state", None), "name", "") == "FAILED":
        raise RuntimeError("Gemini failed to process the video")
    return uploaded


async def run_video_variant(client, path: Path, prompt: str, expected: str) -> list[dict[str, Any]]:
    runs = []
    config = generation_config()
    for index in range(1, RUNS + 1):
        end_to_end_started = time.perf_counter()
        uploaded = None
        try:
            uploaded = await asyncio.to_thread(client.files.upload, file=path)
            uploaded = await wait_for_file(client, uploaded)
            request_started = time.perf_counter()
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL, contents=[prompt, uploaded], config=config,
            )
            request_ms = (time.perf_counter() - request_started) * 1000
            run = classify(response.text or "", expected)
            run.update({
                "run": index,
                "gemini_request_latency_ms": request_ms,
                "end_to_end_latency_ms": (time.perf_counter() - end_to_end_started) * 1000,
                "usage": serializable(getattr(response, "usage_metadata", None)),
            })
        except Exception as exc:
            run = {
                "run": index,
                "end_to_end_latency_ms": (time.perf_counter() - end_to_end_started) * 1000,
                "api_error": str(exc), "card_correct": False, "validated_correct": False,
            }
        finally:
            if uploaded is not None and getattr(uploaded, "name", None):
                try:
                    await asyncio.to_thread(client.files.delete, name=uploaded.name)
                except Exception:
                    pass
        runs.append(run)
    return runs


async def run_image_variant(client, paths: list[Path], prompt: str, expected: str) -> list[dict[str, Any]]:
    from google.genai import types
    frames = [types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg") for path in paths]
    config = generation_config()
    runs = []
    for index in range(1, RUNS + 1):
        end_to_end_started = time.perf_counter()
        try:
            request_started = time.perf_counter()
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL, contents=[prompt, *frames], config=config,
            )
            request_ms = (time.perf_counter() - request_started) * 1000
            run = classify(response.text or "", expected)
            run.update({
                "run": index,
                "gemini_request_latency_ms": request_ms,
                "end_to_end_latency_ms": (time.perf_counter() - end_to_end_started) * 1000,
                "usage": serializable(getattr(response, "usage_metadata", None)),
            })
        except Exception as exc:
            run = {
                "run": index,
                "end_to_end_latency_ms": (time.perf_counter() - end_to_end_started) * 1000,
                "api_error": str(exc), "card_correct": False, "validated_correct": False,
            }
        runs.append(run)
    return runs


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if "api_error" not in run]
    def average(key: str):
        values = [float(run[key]) for run in completed if key in run]
        return sum(values) / len(values) if values else None
    detections = [run.get("detected_result") for run in completed]
    return {
        "card_accuracy": sum(run.get("card_correct") is True for run in runs) / len(runs),
        "card_correct_runs": sum(run.get("card_correct") is True for run in runs),
        "validated_accuracy": sum(run.get("validated_correct") is True for run in runs) / len(runs),
        "validated_correct_runs": sum(run.get("validated_correct") is True for run in runs),
        "average_end_to_end_latency_ms": average("end_to_end_latency_ms"),
        "average_gemini_request_latency_ms": average("gemini_request_latency_ms"),
        "consistent": bool(detections) and len(set(detections)) == 1,
        "detected_results": detections,
    }


def include_preprocessing(variant: dict[str, Any]) -> None:
    preprocessing_ms = float(variant["preprocessing_ms"])
    for run in variant["runs"]:
        if "end_to_end_latency_ms" in run:
            media_latency = float(run["end_to_end_latency_ms"])
            run["media_submission_and_request_latency_ms"] = media_latency
            run["end_to_end_latency_ms"] = preprocessing_ms + media_latency


async def benchmark(expected: str) -> int:
    load_dotenv(BACKEND_DIR / ".env")
    if not CLIP_PATH.is_file():
        raise FileNotFoundError(CLIP_PATH)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    from google import genai
    client = genai.Client(api_key=api_key)  # Deliberately reused for all nine requests.
    prompt = played_card_prompt()

    with tempfile.TemporaryDirectory(prefix="programat-gemini-latency-") as directory:
        work = Path(directory)
        # The production streamer may overwrite last_hosted_clip.mp4. Freeze its
        # bytes once so every derived representation and all nine requests use
        # the same source artifact.
        frozen_full_video = work / "full-6s-1080x1920.mp4"
        shutil.copyfile(CLIP_PATH, frozen_full_video)
        frozen_bytes = frozen_full_video.read_bytes()
        frozen_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
        short_video = work / "four-seconds-720x1280.mp4"
        short_preprocess_ms = prepare_four_second_video(frozen_full_video, short_video)
        jpeg_paths, jpeg_preprocess_ms = prepare_jpegs(frozen_full_video, work / "jpegs", 8)
        variants = {
            "full_6s_mp4_1080x1920": {
                "source_path": str(CLIP_PATH), "sha256": frozen_sha256,
                "input_bytes": len(frozen_bytes),
                "preprocessing_ms": 0.0,
                "runs": await run_video_variant(client, frozen_full_video, prompt, expected),
            },
            "four_second_mp4_720x1280": {
                "path": str(short_video), "input_bytes": short_video.stat().st_size,
                "preprocessing_ms": short_preprocess_ms,
                "runs": await run_video_variant(client, short_video, prompt, expected),
            },
            "eight_ordered_jpegs_720x1280": {
                "frame_count": len(jpeg_paths),
                "input_bytes": sum(path.stat().st_size for path in jpeg_paths),
                "preprocessing_ms": jpeg_preprocess_ms,
                "runs": await run_image_variant(client, jpeg_paths, prompt, expected),
            },
        }
    for variant in variants.values():
        include_preprocessing(variant)
        variant["summary"] = summarize(variant["runs"])
    report = {
        "clip_path": str(CLIP_PATH), "frozen_clip_sha256": frozen_sha256,
        "frozen_clip_bytes": len(frozen_bytes), "model": MODEL, "expected_result": expected,
        "prompt": prompt, "runs_per_variant": RUNS,
        "temperature": 0, "structured_json": True,
        "automatic_function_calling_disabled": True, "variants": variants,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, variant in variants.items():
        print(f"\nvariant={name}")
        print(json.dumps(variant, indent=2, ensure_ascii=False))
    print(f"\nreport_path={REPORT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-card")
    expected.add_argument("--expected-no-event", action="store_true")
    args = parser.parse_args()
    truth = "no_event" if args.expected_no_event else normalize_expected_card(args.expected_card)
    try:
        return asyncio.run(benchmark(truth))
    except (FileNotFoundError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"benchmark_failed={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
