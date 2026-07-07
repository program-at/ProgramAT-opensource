"""Shared, streaming-independent Moondream Cloud request helpers."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping

BASE_URL = "https://api.moondream.ai/v1"
MESSAGE_FORMAT = "nested_image_url_url"
SHORT_GOAL_MAX_CHARS = 160
PROMPT_MAX_CHARS = 250


def input_type(image: Any) -> str:
    if isinstance(image, str) and image.startswith("data:image"):
        return "data_uri"
    if isinstance(image, (bytes, bytearray)):
        return "bytes"
    from PIL import Image
    import numpy as np
    if isinstance(image, np.ndarray):
        return "ndarray"
    if isinstance(image, Image.Image):
        return "PIL"
    return type(image).__name__


def image_bytes(image: Any) -> bytes:
    if isinstance(image, str):
        payload = image.split(",", 1)[1] if image.startswith("data:image") and "," in image else image
        return base64.b64decode(payload)
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    from PIL import Image
    import numpy as np
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            pil_image = Image.fromarray(array, mode="L").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 1:
            pil_image = Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 3:
            # stream_server.decode_frame() returns OpenCV BGR arrays.
            pil_image = Image.fromarray(array[:, :, ::-1].copy(), mode="RGB")
        elif array.ndim == 3 and array.shape[2] == 4:
            # Preserve alpha long enough for Pillow to composite via RGB conversion.
            pil_image = Image.fromarray(array[:, :, [2, 1, 0, 3]].copy(), mode="RGBA").convert("RGB")
        else:
            raise TypeError(f"Unsupported ndarray image shape: {array.shape}")
        output = io.BytesIO()
        pil_image.save(output, format="JPEG")
        return output.getvalue()
    if isinstance(image, Image.Image):
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG")
        return output.getvalue()
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def prepare_image(image: Any) -> tuple[bytes, str, tuple[int, int]]:
    from PIL import Image, ImageOps

    max_dimension = max(1, int(os.environ.get("MOONDREAM_MAX_IMAGE_DIMENSION", "768")))
    quality = min(95, max(1, int(os.environ.get("MOONDREAM_JPEG_QUALITY", "75"))))
    with Image.open(io.BytesIO(image_bytes(image))) as opened:
        normalized = ImageOps.exif_transpose(opened).convert("RGB")
        normalized.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        dimensions = normalized.size
        output = io.BytesIO()
        normalized.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue(), "image/jpeg", dimensions


def build_messages(
    messages: Iterable[Mapping[str, Any]], image_data_uri: str
) -> List[Dict[str, Any]]:
    request_messages = [dict(message) for message in messages]
    image_part = {"type": "image_url", "image_url": {"url": image_data_uri}}
    user_index = next(
        (index for index in range(len(request_messages) - 1, -1, -1)
         if request_messages[index].get("role") == "user"), None,
    )
    if user_index is None:
        request_messages.append({"role": "user", "content": [image_part]})
    else:
        content = request_messages[user_index].get("content", "")
        text_parts = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        text_parts = [
            part for part in text_parts
            if not isinstance(part, dict) or part.get("type") != "image_url"
        ]
        # Documented production default: text first, then nested image_url.url.
        request_messages[user_index]["content"] = [*text_parts, image_part]
    return request_messages


def prompt_text(messages: Iterable[Mapping[str, Any]]) -> str:
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text") or "").strip()
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
    return ""


def all_prompt_text(messages: Iterable[Mapping[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return "\n".join(part for part in parts if part)


def normalize_short_goal(value: Any, max_chars: int = SHORT_GOAL_MAX_CHARS) -> str:
    text = str(value or "")
    text = re.sub(r"```(?:\w+)?", " ", text)
    text = text.replace("`", " ").replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*(?:#{1,6}|[-*+] |\d+[.)] )", "", text)
    boilerplate_patterns = (
        r"return only concise.*?(?:\.|$)",
        r"do not use markdown.*?(?:\.|$)",
        r"output (?:exactly|only).*?(?:\.|$)",
        r"useful previous-stage artifact.*?(?:\.|$)",
        r"you are responding to a live camera stream.*?(?:\.|$)",
        r"you provide navigation guidance.*?(?:\.|$)",
    )
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = " ".join(text.split()).strip(" -:;,.")
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened or text[:max_chars]


def adapt_task_prompt(
    messages: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None
) -> tuple[str, str, int]:
    """Return Moondream's short prompt, normalized goal, and original prompt length."""
    message_list = list(messages)
    metadata = metadata or {}
    original = all_prompt_text(message_list)
    raw_goal = next(
        (
            metadata.get(key)
            for key in ("stage_goal", "goal", "task_text", "task")
            if metadata.get(key)
        ),
        None,
    )
    short_goal = normalize_short_goal(raw_goal or prompt_text(message_list))
    if not short_goal:
        short_goal = "Describe the relevant visual information"
    capability = str(metadata.get("capability") or "").strip().lower()
    goal_lower = short_goal.lower()
    if capability == "ocr" or any(
        word in goal_lower for word in ("read", "text", "document", "mail", "envelope", "letter")
    ):
        template = "Read visible text and briefly answer this task: {}"
    elif capability in {"navigation", "spatial_reasoning", "object_detection_localization"} or any(
        word in goal_lower for word in ("navigate", "location", "locate", "where", "exit")
    ):
        template = "Briefly identify relevant objects and locations for this task: {}"
    elif capability == "general_reasoning":
        template = "Answer this visual question briefly: {}"
    elif capability == "structured_visual_understanding":
        template = "Use the image to answer briefly: {}"
    else:
        template = "Briefly answer the visual task using the image: {}"
    adapted = template.format(short_goal)
    if len(adapted) > PROMPT_MAX_CHARS:
        adapted = adapted[:PROMPT_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return adapted, short_goal, len(original)


def create_client(api_key: str, timeout: float):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout, max_retries=0)


def response_text(completion: Any) -> str:
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return ""


def error_context(exc: Exception) -> tuple[Any, str, str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if request_id is None and response is not None:
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
    body = getattr(response, "text", None) or getattr(exc, "body", None) or str(exc)
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False, default=str)
    return status, request_id or "unavailable", body
