#!/usr/bin/env python3
"""Standalone hosted NVIDIA MP4 capability probe; does not start ProgramAT."""

from __future__ import annotations

import argparse
import asyncio
import ast
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from nvidia_hosted_client import (  # noqa: E402
    NvidiaHostedClient,
    NvidiaHostedError,
    build_video_request,
    parse_structured_video_result,
    validate_played_card_event,
)


def make_sample(path: Path, seconds: float, fps: float) -> None:
    command = [
        os.getenv("PROGRAMAT_FFMPEG_BINARY", "ffmpeg"), "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate={fps}", "-t", str(seconds),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-y", str(path),
    ]
    subprocess.run(command, check=True)


def played_card_prompt() -> str:
    path = BACKEND_DIR.parent / "tools" / "played_card_rtvi.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TOOL_PROMPT"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise NvidiaHostedError("played_card_rtvi has no literal TOOL_PROMPT")


async def run(video_path: Path | None, prompt: str, validate_cards: bool) -> int:
    load_dotenv(BACKEND_DIR / ".env")
    base_url = os.getenv("NVIDIA_VIDEO_BASE_URL") or os.getenv(
        "NVIDIA_HOSTED_API_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    api_key = os.getenv("NVIDIA_VIDEO_API_KEY") or os.getenv("NVIDIA_HOSTED_API_KEY", "")
    model = os.getenv("NVIDIA_VIDEO_MODEL") or os.getenv("NVIDIA_HOSTED_MODEL", "")
    input_mode = os.getenv("NVIDIA_VIDEO_INPUT_MODE", "base64").strip().lower()
    if input_mode != "base64":
        raise NvidiaHostedError(
            "This endpoint probe supports only the explicitly configured base64 mode; "
            "integrate.api.nvidia.com does not expose an assumed /files upload API."
        )
    if not api_key or not model:
        raise NvidiaHostedError("NVIDIA video API key and model are required")

    with tempfile.TemporaryDirectory(prefix="programat-video-probe-") as temp_dir:
        path = video_path or Path(temp_dir) / "sample.mp4"
        encode_started = time.perf_counter()
        if video_path is None:
            make_sample(path, 2, 4)
        encode_ms = (time.perf_counter() - encode_started) * 1000
        video = path.read_bytes()
        limit = int(os.getenv("HOSTED_VIDEO_MAX_CLIP_BYTES", "8388608") or "8388608")
        if len(video) > limit:
            raise NvidiaHostedError(f"MP4 is {len(video)} bytes; base64 limit is {limit}")
        print(f"clip={path} bytes={len(video)} input_mode=base64 model={model} encode_ms={encode_ms:.1f}")
        data_uri = "data:video/mp4;base64," + base64.b64encode(video).decode("ascii")
        request = build_video_request(model, prompt, data_uri, 80)
        client = NvidiaHostedClient(base_url, api_key, 60)
        try:
            inference_started = time.perf_counter()
            response = await client.chat_completions(request)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            print(f"raw_model_response={response}")
            if validate_cards:
                try:
                    parsed = parse_structured_video_result(response)
                    emitted, decision = validate_played_card_event(parsed, 0.8)
                    print(f"parsed_result={json.dumps(parsed, ensure_ascii=False, sort_keys=True)}")
                    print(f"validation_decision={decision} emitted_message={emitted!r}")
                except NvidiaHostedError as exc:
                    print(f"validation_decision=invalid_json error={exc}")
            print(f"inference_ms={inference_ms:.1f} total_ms={encode_ms + inference_ms:.1f}")
            return 0
        finally:
            await client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--played-card", action="store_true", help="Use and validate played_card_rtvi JSON output")
    args = parser.parse_args()
    if args.video is not None and not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    try:
        prompt = args.prompt or (
            played_card_prompt() if args.played_card
            else "Describe this chronological test video briefly."
        )
        return asyncio.run(run(args.video, prompt, args.played_card))
    except (NvidiaHostedError, subprocess.CalledProcessError, OSError) as exc:
        print(f"probe_failed={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
