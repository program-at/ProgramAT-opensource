#!/usr/bin/env python3
"""Compare NVIDIA video models on one saved played-card MP4 without changing production."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
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
from scripts.test_nvidia_hosted_video import played_card_prompt  # noqa: E402


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


async def compare(clip: Path, expected_card: str | None) -> int:
    load_dotenv(BACKEND_DIR / ".env")
    base_url = os.getenv("NVIDIA_VIDEO_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key = os.getenv("NVIDIA_VIDEO_API_KEY", "").strip()
    production = os.getenv("NVIDIA_VIDEO_MODEL", "").strip()
    if not api_key or not production:
        raise NvidiaHostedError("NVIDIA_VIDEO_API_KEY and NVIDIA_VIDEO_MODEL are required")

    client = NvidiaHostedClient(base_url, api_key, 60)
    try:
        available = await client.list_models()
        by_id = {model.id: model for model in available}
        qwen = [
            model.id for model in available
            if all(marker in model.id.casefold() for marker in ("qwen3", "vl", "30b", "a3b", "instruct"))
        ]
        configured = [
            item for item in os.getenv("NVIDIA_VIDEO_COMPARISON_MODELS", "").split(",")
            if item.strip()
        ]
        advertised_video = [model.id for model in available if model.supports_video]
        candidates = _unique([production, *qwen, *configured, *advertised_video])
        candidates = [model for model in candidates if model in by_id]
        print("available_candidates=" + json.dumps(candidates))

        raw_video = clip.read_bytes()
        limit = int(os.getenv("HOSTED_VIDEO_MAX_CLIP_BYTES", "8388608"))
        if len(raw_video) > limit:
            raise NvidiaHostedError(f"MP4 is {len(raw_video)} bytes; limit is {limit}")
        data_uri = "data:video/mp4;base64," + base64.b64encode(raw_video).decode("ascii")
        prompt = played_card_prompt()
        max_tokens = int(os.getenv("HOSTED_VIDEO_MAX_TOKENS", "256"))
        expected = " ".join(expected_card.casefold().split()) if expected_card else None

        for model in candidates:
            started = time.perf_counter()
            try:
                raw = await client.chat_completions(
                    build_video_request(model, prompt, data_uri, max_tokens)
                )
                latency_ms = (time.perf_counter() - started) * 1000
                print(f"\nmodel={model}\nlatency_ms={latency_ms:.1f}\nraw_output={raw}")
                try:
                    parsed = parse_structured_video_result(raw)
                    emitted, decision = validate_played_card_event(parsed, 0.8)
                    parsed_card = parsed.get("played_card")
                    correctness = (
                        "unknown" if expected is None
                        else str(
                            decision == "accepted"
                            and isinstance(parsed_card, str)
                            and " ".join(parsed_card.casefold().split()) == expected
                        ).lower()
                    )
                    print("parsed_cards=" + json.dumps({
                        "before_cards": parsed.get("before_cards"),
                        "after_cards": parsed.get("after_cards"),
                        "played_card": parsed_card,
                    }, ensure_ascii=False, sort_keys=True))
                    print(f"validation={decision} correctness={correctness} emitted={emitted!r}")
                except NvidiaHostedError as exc:
                    print(f"parsed_cards=null validation=invalid_json correctness=false error={exc}")
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                print(f"\nmodel={model}\nlatency_ms={latency_ms:.1f}\nrequest_error={exc}")
        return 0
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--expected-card", help="Canonical ground truth, e.g. 'jack of diamonds'")
    args = parser.parse_args()
    if not args.clip.is_file():
        parser.error(f"Clip does not exist: {args.clip}")
    try:
        return asyncio.run(compare(args.clip, args.expected_card))
    except (NvidiaHostedError, OSError) as exc:
        print(f"comparison_failed={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
