#!/usr/bin/env python3
"""Benchmark VLM latency on a single image for non-Qwen models.

This script measures end-to-end latency for VLMs listed in model_profiles.yaml,
excluding any model whose name or model id contains "qwen".

Supported execution modes:
- Remote/API models: use the existing LiteLLM call path.
- Local open-source models: use Transformers for LLaVA-OneVision and LLaVA-Video.

The script writes a JSON report with timings and a short response preview for each model.
You can also run grouped benchmarks with --group:
- all: all non-Qwen VLMs
- gemini: only Gemini models
- others: non-Gemini models
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
from model_router import load_model_profiles


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BACKEND_DIR / "latency_test_image.jpg"
DEFAULT_OUT = BACKEND_DIR / "vlm_latency_results.json"
DEFAULT_PROMPT = "Briefly describe the image and mention the main objects or text."

LOCAL_TRANSFORMER_MODELS = {
    "LLaVA-OneVision-7B": "lmms-lab/llava-onevision-qwen2-7b-ov",
    "LLaVA-Video-7B": "lmms-lab/LLaVA-Video-7B-Qwen2",
}


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


def _is_qwen_profile(profile: Any) -> bool:
    # Exclude explicit Qwen profile entries only.
    # Some non-Qwen profiles (e.g. LLaVA variants) may contain "qwen" in their
    # upstream model id but should still be benchmarked.
    name = _normalize_name(getattr(profile, "name", ""))
    return name.startswith("qwen")


def _is_gemini_profile(profile: Any) -> bool:
    name = _normalize_name(getattr(profile, "name", ""))
    model = _normalize_name(getattr(profile, "model", ""))
    return name.startswith("gemini") or model.startswith("gemini/")


def _select_profiles(group: Literal["all", "gemini", "others"] = "all") -> List[Any]:
    profiles = [profile for profile in load_model_profiles() if getattr(profile, "type", "") == "general_vlm"]
    profiles = [profile for profile in profiles if not _is_qwen_profile(profile)]
    if group == "gemini":
        return [profile for profile in profiles if _is_gemini_profile(profile)]
    if group == "others":
        return [profile for profile in profiles if not _is_gemini_profile(profile)]
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


def _build_transformers_messages(prompt: str, image: Image.Image) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _benchmark_local_model(profile: Any, image: Image.Image, prompt: str, model_id: str) -> BenchmarkResult:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        return BenchmarkResult(
            profile_name=profile.name,
            profile_model=model_id,
            backend="transformers",
            status="error",
            load_time_s=None,
            inference_time_s=None,
            total_time_s=None,
            response_text="",
            error=f"Transformers backend unavailable: {exc}",
        )

    load_started = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        model.eval()
        load_time_s = time.perf_counter() - load_started

        inference_started = time.perf_counter()
        messages = _build_transformers_messages(prompt, image)

        try:
            chat_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(
                text=chat_text,
                images=image,
                return_tensors="pt",
            )
        except Exception:
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

        target_device = getattr(model, "device", None)
        if target_device is None:
            target_device = next(model.parameters()).device

        if hasattr(inputs, "to"):
            inputs = inputs.to(target_device)
        else:
            inputs = {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96)

        inference_time_s = time.perf_counter() - inference_started
        token_start = inputs["input_ids"].shape[-1] if isinstance(inputs, dict) and "input_ids" in inputs else 0
        try:
            response_text = processor.decode(outputs[0][token_start:], skip_special_tokens=True)
        except Exception:
            response_text = processor.decode(outputs[0], skip_special_tokens=True)

        total_time_s = time.perf_counter() - load_started
        return BenchmarkResult(
            profile_name=profile.name,
            profile_model=model_id,
            backend="transformers",
            status="ok",
            load_time_s=round(load_time_s, 4),
            inference_time_s=round(inference_time_s, 4),
            total_time_s=round(total_time_s, 4),
            response_text=response_text,
        )
    except Exception as exc:
        total_time_s = time.perf_counter() - load_started
        return BenchmarkResult(
            profile_name=profile.name,
            profile_model=model_id,
            backend="transformers",
            status="error",
            load_time_s=round(time.perf_counter() - load_started, 4),
            inference_time_s=None,
            total_time_s=round(total_time_s, 4),
            response_text="",
            error=str(exc),
        )


def benchmark_model(profile: Any, image: Image.Image, prompt: str) -> BenchmarkResult:
    if profile.name in LOCAL_TRANSFORMER_MODELS:
        return _benchmark_local_model(profile, image, prompt, LOCAL_TRANSFORMER_MODELS[profile.name])
    return _benchmark_api_model(profile, image, prompt)


def run_benchmark(image_path: Path, prompt: str, group: Literal["all", "gemini", "others"] = "all") -> Dict[str, Any]:
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
        choices=["all", "gemini", "others"],
        default="all",
        help="Model group to benchmark",
    )
    args = parser.parse_args()

    payload = run_benchmark(args.image, args.prompt, group=args.group)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote results to: {args.out}")


if __name__ == "__main__":
    main()