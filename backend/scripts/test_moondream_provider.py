#!/usr/bin/env python3
"""Exercise Moondream Cloud directly, outside ProgramAT streaming."""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import moondream_provider  # noqa: E402


def tiny_jpeg() -> bytes:
    output = io.BytesIO()
    image = Image.new("RGB", (32, 24), "white")
    image.putpixel((16, 12), (0, 0, 0))
    image.save(output, format="JPEG", quality=75, optimize=True)
    payload = output.getvalue()
    with Image.open(io.BytesIO(payload)) as check:
        check.verify()
    return payload


def raw_response(completion: object) -> str:
    dump = getattr(completion, "model_dump_json", None)
    return dump(indent=2) if callable(dump) else repr(completion)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional local image instead of the tiny fixture")
    parser.add_argument("--model", help="Override MOONDREAM_MODEL")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--capability", default="general_reasoning")
    parser.add_argument("--prompt", default="Describe this image briefly.")
    args = parser.parse_args()

    load_dotenv(BACKEND_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api_key = os.environ.get("MOONDREAM_API_KEY", "").strip()
    if not api_key:
        parser.error("MOONDREAM_API_KEY is not configured")
    model = args.model or os.environ.get("MOONDREAM_MODEL", "moondream/moondream3-preview")
    source = args.image.read_bytes() if args.image else tiny_jpeg()
    payload, mime_type, dimensions = moondream_provider.prepare_image(source)
    data_uri = f"data:{mime_type};base64," + base64.b64encode(payload).decode("ascii")
    adapted_prompt, _short_task, _original_chars, _source = moondream_provider.adapt_task_prompt(
        [{"role": "user", "content": args.prompt}],
        {"capability": args.capability, "goal": args.prompt},
    )
    messages = moondream_provider.build_messages(
        [{"role": "user", "content": adapted_prompt}], data_uri
    )
    logging.info(
        "[Moondream Smoke] model=%s timeout_seconds=%.1f dimensions=%sx%s "
        "payload_bytes=%d prompt_chars=%d",
        model, args.timeout, dimensions[0], dimensions[1], len(payload), len(adapted_prompt),
    )
    started = time.perf_counter()
    try:
        completion = moondream_provider.create_client(api_key, args.timeout).chat.completions.create(
            model=model, messages=messages
        )
        print(raw_response(completion))
        text = moondream_provider.response_text(completion)
        logging.info(
            "[Moondream Smoke] status=200 request_id=%s latency_ms=%.1f text_chars=%d",
            getattr(completion, "_request_id", None) or "unavailable",
            (time.perf_counter() - started) * 1000, len(text),
        )
        return 0 if text else 1
    except Exception as exc:
        status, request_id, body = moondream_provider.error_context(exc)
        logging.error(
            "[Moondream Smoke] status=%s request_id=%s latency_ms=%.1f error=%s",
            status, request_id, (time.perf_counter() - started) * 1000, body,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
