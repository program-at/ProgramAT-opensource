#!/usr/bin/env python3
"""Run the exact last hosted MP4 three times on Nemotron and Gemini."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from nvidia_hosted_client import (  # noqa: E402
    NvidiaHostedError,
    build_video_request,
    extract_chat_content,
    parse_structured_video_result,
    validate_played_card_event,
)
from scripts.test_nvidia_hosted_video import played_card_prompt  # noqa: E402

CLIP_PATH = BACKEND_DIR / "debug" / "last_hosted_clip.mp4"
REPORT_PATH = BACKEND_DIR / "debug" / "last_hosted_clip_comparison.json"
NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl"


def normalize_expected_card(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    words = normalized.split()
    if len(words) == 3 and words[1] == "of" and words[2] in {
        "club", "diamond", "heart", "spade",
    }:
        words[2] += "s"
    return " ".join(words)


def configured_gemini_model() -> str:
    policy = yaml.safe_load((BACKEND_DIR / "execution_policy.yaml").read_text())
    model = policy["implementations"]["gemini_flash_lite"]["model"]
    if not isinstance(model, str) or not model.startswith("gemini/"):
        raise RuntimeError("gemini_flash_lite is not configured as a Gemini model")
    return model


def serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if hasattr(value, "model_dump"):
        return serializable(value.model_dump(exclude_none=True))
    if hasattr(value, "to_dict"):
        return serializable(value.to_dict())
    return str(value)


def interpret(raw: str, expected: str) -> dict[str, Any]:
    result: dict[str, Any] = {"raw_response": raw}
    try:
        parsed = parse_structured_video_result(raw)
        emitted, validation = validate_played_card_event(parsed, 0.8)
        if validation == "accepted":
            detected = normalize_expected_card(str(parsed["played_card"]))
        elif validation == "stable_no_event":
            detected = "no_event"
        else:
            detected = "rejected"
        result.update({
            "parsed_json": parsed,
            "validation": validation,
            "detected_result": detected,
            "emitted_message": emitted,
            "correct": detected == expected,
        })
    except NvidiaHostedError as exc:
        result.update({
            "parsed_json": None,
            "validation": "invalid_json",
            "detected_result": "rejected",
            "emitted_message": None,
            "correct": False,
            "parse_error": str(exc),
        })
    return result


async def run_nemotron(video: bytes, prompt: str, expected: str) -> list[dict[str, Any]]:
    base_url = os.getenv("NVIDIA_VIDEO_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    api_key = os.getenv("NVIDIA_VIDEO_API_KEY", "").strip()
    if not api_key:
        return [{"api_error": "NVIDIA_VIDEO_API_KEY is not configured"} for _ in range(3)]
    uri = "data:video/mp4;base64," + base64.b64encode(video).decode("ascii")
    request = build_video_request(NEMOTRON_MODEL, prompt, uri, 256)
    headers = {"Authorization": f"Bearer {api_key}"}
    runs = []
    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        for index in range(1, 4):
            started = time.perf_counter()
            try:
                async with session.post(f"{base_url}/chat/completions", json=request) as response:
                    body = await response.text()
                    latency = (time.perf_counter() - started) * 1000
                    if response.status >= 400:
                        run = {"run": index, "latency_ms": latency, "api_error": body}
                    else:
                        payload = json.loads(body)
                        run = interpret(extract_chat_content(payload), expected)
                        run.update({
                            "run": index,
                            "latency_ms": latency,
                            "usage": serializable(payload.get("usage")),
                        })
            except Exception as exc:
                run = {
                    "run": index,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "api_error": str(exc),
                }
            runs.append(run)
            print_run(NEMOTRON_MODEL, run)
    return runs


async def run_gemini(video_path: Path, prompt: str, expected: str, model: str) -> list[dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return [{"api_error": "GEMINI_API_KEY is not configured"} for _ in range(3)]
    from google import genai

    client = genai.Client(api_key=api_key)
    uploaded = None
    runs: list[dict[str, Any]] = []
    try:
        uploaded = await asyncio.to_thread(client.files.upload, file=video_path)
        deadline = time.monotonic() + 120
        while str(getattr(uploaded, "state", "")).upper().endswith("PROCESSING"):
            if time.monotonic() >= deadline:
                raise TimeoutError("Gemini video processing did not finish within 120 seconds")
            await asyncio.sleep(2)
            uploaded = await asyncio.to_thread(client.files.get, name=uploaded.name)
        for index in range(1, 4):
            started = time.perf_counter()
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model.removeprefix("gemini/"),
                    contents=[prompt, uploaded],
                )
                run = interpret(response.text or "", expected)
                run.update({
                    "run": index,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "usage": serializable(getattr(response, "usage_metadata", None)),
                })
            except Exception as exc:
                run = {
                    "run": index,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "api_error": str(exc),
                }
            runs.append(run)
            print_run(model, run)
    except Exception as exc:
        runs = [{"run": index, "api_error": str(exc)} for index in range(1, 4)]
        for run in runs:
            print_run(model, run)
    finally:
        if uploaded is not None and getattr(uploaded, "name", None):
            try:
                await asyncio.to_thread(client.files.delete, name=uploaded.name)
            except Exception:
                pass
    return runs


def print_run(model: str, run: dict[str, Any]) -> None:
    print(f"\nmodel={model} run={run.get('run')}")
    for key in ("raw_response", "parsed_json", "detected_result", "latency_ms", "usage", "api_error"):
        if key in run:
            print(f"{key}={json.dumps(run[key], ensure_ascii=False, default=str)}")


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if "api_error" not in run]
    latencies = [float(run["latency_ms"]) for run in successful if "latency_ms" in run]
    detections = [run.get("detected_result") for run in successful]
    return {
        "video_accepted": bool(successful),
        "correct_runs": sum(run.get("correct") is True for run in successful),
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "consistent": bool(detections) and len(set(detections)) == 1,
        "detected_results": detections,
    }


async def main_async(expected: str) -> int:
    load_dotenv(BACKEND_DIR / ".env")
    if not CLIP_PATH.is_file():
        raise FileNotFoundError(
            f"{CLIP_PATH} does not exist; run one hosted played-card inference first"
        )
    video = CLIP_PATH.read_bytes()
    prompt = played_card_prompt()
    gemini_model = configured_gemini_model()
    nemotron = await run_nemotron(video, prompt, expected)
    gemini = await run_gemini(CLIP_PATH, prompt, expected, gemini_model)
    report = {
        "clip_path": str(CLIP_PATH),
        "clip_bytes": len(video),
        "prompt": prompt,
        "expected_result": expected,
        "runs_per_model": 3,
        "models": {
            NEMOTRON_MODEL: {"runs": nemotron, "summary": summarize(nemotron)},
            gemini_model: {"runs": gemini, "summary": summarize(gemini)},
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport_path={REPORT_PATH}")
    print(json.dumps({model: data["summary"] for model, data in report["models"].items()}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-card")
    expected.add_argument("--expected-no-event", action="store_true")
    args = parser.parse_args()
    truth = "no_event" if args.expected_no_event else normalize_expected_card(args.expected_card)
    try:
        return asyncio.run(main_async(truth))
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"comparison_failed={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
