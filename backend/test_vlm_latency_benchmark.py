#!/usr/bin/env python3
"""Benchmark configured VLM implementation latency on a single image.

This script measures end-to-end latency for model implementations listed in
execution_policy.yaml through the same LiteLLM call path used at runtime.

The script writes a JSON report with timings and a short response preview for each model.
You can also run grouped benchmarks with --group:
- all: every configured VLM
- groq: Groq implementations
- dashscope: DashScope implementations
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

import model_router
from litellm_utils import extract_text
from model_router import load_implementation_profiles


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BACKEND_DIR / "latency_test_image.jpg"
DEFAULT_OUT = BACKEND_DIR / "vlm_latency_results.json"
DEFAULT_PROMPT = "Briefly describe the image and mention the main objects or text."

@dataclass
class BenchmarkResult:
    profile_name: str
    profile_model: str
    backend: str
    status: str
    load_time_s: Optional[float]
    inference_time_s: Optional[float]
    total_time_s: Optional[float]
    response_text: str
    error: str = ""


def _normalize_name(value: str) -> str:
    return str(value or "").strip().lower()


def _provider(profile: Any) -> str:
    model = _normalize_name(getattr(profile, "model", ""))
    return model.split("/", 1)[0]


def _select_profiles(group: Literal["all", "groq", "dashscope"] = "all") -> List[Any]:
    profiles = [profile for profile in load_implementation_profiles().values() if profile.kind == "model"]
    if group != "all":
        return [profile for profile in profiles if _provider(profile) == group]
    return profiles


def _load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def _short_preview(text: str, max_chars: int = 300) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _benchmark_api_model(profile: Any, image: Image.Image, prompt: str) -> BenchmarkResult:
    started = time.perf_counter()
    try:
        response = model_router.call_model(
            profile.model,
            [{"role": "user", "content": prompt}],
            images=[image],
            metadata={"temperature": 0, "max_tokens": 96},
        )
        inference_time_s = time.perf_counter() - started
        response_text = extract_text(response)
        total_time_s = time.perf_counter() - started
        return BenchmarkResult(
            profile_name=profile.name,
            profile_model=profile.model,
            backend="api",
            status="ok",
            load_time_s=None,
            inference_time_s=round(inference_time_s, 4),
            total_time_s=round(total_time_s, 4),
            response_text=response_text,
        )
    except Exception as exc:
        total_time_s = time.perf_counter() - started
        return BenchmarkResult(
            profile_name=profile.name,
            profile_model=profile.model,
            backend="api",
            status="error",
            load_time_s=None,
            inference_time_s=None,
            total_time_s=round(total_time_s, 4),
            response_text="",
            error=str(exc),
        )


def benchmark_model(profile: Any, image: Image.Image, prompt: str) -> BenchmarkResult:
    return _benchmark_api_model(profile, image, prompt)


def run_benchmark(
    image_path: Path,
    prompt: str,
    group: Literal["all", "groq", "dashscope"] = "all",
) -> Dict[str, Any]:
    image = _load_image(image_path)
    profiles = _select_profiles(group=group)
    if not profiles:
        raise RuntimeError(f"No matching models found for group='{group}'")
    results = []
    for profile in profiles:
        print(f"Benchmarking {profile.name} ({profile.model})...")
        result = benchmark_model(profile, image, prompt)
        results.append(asdict(result))
        print(
            f"  -> {result.status} | backend={result.backend} | total={result.total_time_s} s | "
            f"load={result.load_time_s} s | infer={result.inference_time_s} s"
        )

    return {
        "image_path": str(image_path),
        "prompt": prompt,
        "group": group,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to the image to benchmark",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt to send to each model",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["all", "groq", "dashscope"],
        default="all",
        help="Model group to benchmark",
    )
    args = parser.parse_args()

    payload = run_benchmark(args.image, args.prompt, group=args.group)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote results to: {args.out}")


if __name__ == "__main__":
    main()
