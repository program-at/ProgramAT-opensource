#!/usr/bin/env python3
"""Standalone NVIDIA hosted VLM single-image and multi-image probe."""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from nvidia_hosted_client import (  # noqa: E402
    NvidiaHostedClient,
    NvidiaHostedError,
    TimedFrame,
    build_multi_image_request,
    select_hosted_model,
)


def image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


async def run(paths: list[Path], prompt: str) -> int:
    load_dotenv(BACKEND_DIR / ".env")
    client = NvidiaHostedClient(
        os.getenv("NVIDIA_HOSTED_API_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        os.getenv("NVIDIA_HOSTED_API_KEY", ""),
        float(os.getenv("NVIDIA_HOSTED_REQUEST_TIMEOUT_SECONDS", "60")),
    )
    try:
        models = await client.list_models()
        print("Available NVIDIA hosted models:")
        for model in models:
            print(
                f"- {model.id} vision={model.supports_vision} video={model.supports_video}"
            )
        configured = os.getenv("NVIDIA_HOSTED_MODEL", "").strip()
        selected = select_hosted_model(models, configured)
        print(f"Selected model: {selected.id}")

        frames = [
            TimedFrame(float(index), image_data_uri(path))
            for index, path in enumerate(paths)
        ]
        max_tokens = int(os.getenv("NVIDIA_HOSTED_MAX_TOKENS", "80"))

        print("\nSingle-image response:")
        single_request = build_multi_image_request(
            selected.id, prompt, frames[:1], max_tokens
        )
        print(await client.chat_completions(single_request))

        print("\nMulti-image response:")
        multi_request = build_multi_image_request(
            selected.id, prompt, frames, max_tokens
        )
        try:
            print(await client.chat_completions(multi_request))
        except NvidiaHostedError as exc:
            # The exception contains the complete bounded HTTP error body. Do not
            # retry with a different request representation in this probe.
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    except NvidiaHostedError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--prompt",
        default="Describe the scene and any change across these chronological frames.",
    )
    args = parser.parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        parser.error(f"Image files do not exist: {', '.join(missing)}")
    return asyncio.run(run(args.images, args.prompt))


if __name__ == "__main__":
    raise SystemExit(main())
