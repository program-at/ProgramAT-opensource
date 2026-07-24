"""Find available seating with tool-owned scene gating and progressive output."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import (
    call_model,
    call_openai_responses_model,
    extract_text,
)

TOOL_NAME = "empty_seat_detection"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Identify genuinely empty chairs, benches, couches, or other usable seats in "
    "this image. Distinguish occupied seats from empty ones. Give a concise count "
    "and body-relative directions such as left, right, or straight ahead. Do not "
    "use colors or other visual-only landmarks as the sole navigation cue. If seat "
    "availability is uncertain or no seating is visible, say so plainly."
)

SCENE_SIMILARITY_THRESHOLD = 0.985
PROGRESSIVE_MODELS = (
    ("moondream/moondream3-preview", "litellm"),
    ("gemini/gemini-3.1-flash-lite-preview", "litellm"),
    ("gpt-5", "openai_responses"),
)


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized appearance embedding for scene continuity."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(512, dtype=np.float32)
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]
    ).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm else histogram


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = SCENE_SIMILARITY_THRESHOLD,
) -> bool:
    return cosine_similarity(reference, current) >= threshold


def _call_selected_model(model_name: str, provider: str, image: np.ndarray) -> str:
    if provider == "openai_responses":
        return call_openai_responses_model(
            model_name,
            TOOL_PROMPT,
            image,
            reasoning_effort="medium",
            timeout=60,
        )
    response = call_model(
        model_name,
        [{"role": "user", "content": TOOL_PROMPT}],
        images=[image],
        metadata={"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def on_take_photo(runtime, image, input_data):
    """Use one explicitly selected default model for a one-shot request."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize state owned by this tool session."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("completed_generation", None)


async def on_frame(runtime, frame):
    """Analyze each distinct scene once and emit model results as they finish."""
    if runtime.is_cancelled():
        return
    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is not None and is_same_scene(anchor, embedding):
        if (
            runtime.get_state("analyzing_generation") == generation
            or runtime.get_state("completed_generation") == generation
        ):
            return
    else:
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("completed_generation", None)

    if runtime.get_state("analyzing_generation") == generation:
        return
    runtime.set_state("analyzing_generation", generation)

    async def run_model(model_name: str, provider: str):
        try:
            text = await asyncio.to_thread(
                _call_selected_model, model_name, provider, frame.image
            )
            return model_name, text, None
        except Exception as exc:
            return model_name, "", str(exc)

    tasks = [
        asyncio.create_task(run_model(model_name, provider))
        for model_name, provider in PROGRESSIVE_MODELS
    ]
    try:
        for completed in asyncio.as_completed(tasks):
            model_name, text, error = await completed
            if (
                runtime.is_cancelled()
                or runtime.get_state("scene_generation") != generation
            ):
                continue
            if text:
                await runtime.emit(
                    text,
                    partial=True,
                    metadata={
                        "model": model_name,
                        "scene_generation": generation,
                        "frame_id": frame.frame_id,
                    },
                )
            elif error:
                await runtime.emit(
                    "",
                    partial=True,
                    metadata={
                        "model": model_name,
                        "error": error,
                        "scene_generation": generation,
                    },
                )
        if (
            not runtime.is_cancelled()
            and runtime.get_state("scene_generation") == generation
        ):
            runtime.set_state("completed_generation", generation)
            await runtime.emit(
                "",
                final=True,
                metadata={
                    "scene_generation": generation,
                    "frame_id": frame.frame_id,
                },
            )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    """Make outstanding scene work stale before the backend cancels its tasks."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
