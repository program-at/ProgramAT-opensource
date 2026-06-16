from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, Iterable, List, Optional

try:
    import litellm
except ImportError:
    litellm = None

try:
    from PIL import Image
except ImportError:
    Image = None


def extract_text(response) -> str:
    """Extract text content from a LiteLLM response object."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get('text', ''))
            elif hasattr(part, 'text'):
                parts.append(getattr(part, 'text', '') or '')
        return ''.join(parts).strip()
    return str(content).strip()


def pil_image_to_data_uri(pil_image: Image.Image, quality: int = 85) -> str:
    """Convert a PIL image to a JPEG base64 data URI."""
    if Image is None:
        raise ImportError("Pillow is required for pil_image_to_data_uri")
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{image_base64}'


def resolve_api_key(model_name: str = "", explicit_api_key: str = "") -> str:
    if explicit_api_key:
        return explicit_api_key

    normalized = (model_name or "").lower()
    if normalized.startswith("openrouter"):
        return os.environ.get("OPENROUTER_API_KEY", "")
    if normalized.startswith("gemini"):
        return os.environ.get("GEMINI_API_KEY", "")
    if normalized.startswith("groq"):
        return os.environ.get("GROQ_API_KEY", "")
    if normalized.startswith("mistral"):
        return os.environ.get("MISTRAL_API_KEY", "")
    if normalized.startswith("anthropic") or normalized.startswith("claude"):
        return os.environ.get("ANTHROPIC_API_KEY", "")
    if normalized.startswith("openai") or normalized.startswith("gpt"):
        return os.environ.get("OPENAI_API_KEY", "")

    return (
        os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("MISTRAL_API_KEY", "")
    )


def _image_to_data_uri(image: Any) -> str:
    if isinstance(image, str):
        raw = image.strip()
        return raw if raw.startswith("data:") else f"data:image/jpeg;base64,{raw}"

    if isinstance(image, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(image)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    if Image is not None and isinstance(image, Image.Image):
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return pil_image_to_data_uri(image)

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(image, np.ndarray):
        if Image is None:
            raise ImportError("Pillow is required to send numpy images to LiteLLM")
        if image.ndim == 2:
            pil_image = Image.fromarray(image)
        elif image.ndim == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image[:, :, ::-1].copy())
        elif image.ndim == 3 and image.shape[2] == 4:
            pil_image = Image.fromarray(image[:, :, [2, 1, 0, 3]].copy())
        else:
            raise TypeError(f"Unsupported ndarray shape for LLM image: {image.shape}")
        return pil_image_to_data_uri(pil_image)

    raise TypeError(f"Unsupported image type for LLM call: {type(image).__name__}")


def _merge_images(messages: List[Dict[str, Any]], images: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    merged = [dict(message) for message in messages]
    image_items = list(images or [])
    if not image_items:
        return merged

    image_parts = [{"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}} for image in image_items]
    target_index = next((index for index in range(len(merged) - 1, -1, -1) if merged[index].get("role") == "user"), None)
    if target_index is None:
        merged.append({"role": "user", "content": image_parts})
        return merged

    content = merged[target_index].get("content")
    if isinstance(content, list):
        merged[target_index]["content"] = list(content) + image_parts
    elif isinstance(content, str) and content.strip():
        merged[target_index]["content"] = [{"type": "text", "text": content}] + image_parts
    else:
        merged[target_index]["content"] = image_parts
    return merged


def _completion_kwargs(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if not isinstance(metadata, dict):
        return kwargs
    for key in ("temperature", "max_tokens", "top_p", "stop", "timeout", "response_format", "stream", "api_base", "base_url"):
        if key in metadata and metadata[key] is not None:
            kwargs["api_base" if key == "base_url" else key] = metadata[key]
    return kwargs


def call_model(
    model_name: str,
    messages: List[Dict[str, Any]],
    images: Optional[Iterable[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Invoke a configured model through LiteLLM."""
    if litellm is None:
        raise ImportError("litellm is required for call_model")

    explicit_key = ""
    if isinstance(metadata, dict):
        explicit_key = metadata.get("api_key") or metadata.get("explicit_api_key") or ""

    return litellm.completion(
        model=model_name,
        messages=_merge_images(messages or [], images),
        api_key=resolve_api_key(model_name, explicit_key if isinstance(explicit_key, str) else ""),
        **_completion_kwargs(metadata),
    )
