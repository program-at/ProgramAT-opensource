"""
Shared LiteLLM helper utilities for backend modules.
"""

import os
import base64
import io
from PIL import Image


def resolve_model_name(model_name: str, default_model: str = 'gemini-2.5-flash-lite') -> str:
    """Normalize model names for LiteLLM provider routing."""

    raw = (model_name or default_model).strip()

    # Already fully qualified provider/model
    known_providers = [
        'openrouter/',
        'gemini/',
        'openai/',
        'anthropic/',
        'ollama/',
        'groq/',
        'vertex_ai/',
    ]

    if any(raw.startswith(p) for p in known_providers):
        return raw

    # Gemini shorthand
    if raw.startswith('gemini'):
        return f'gemini/{raw}'

    # Claude shorthand
    if raw.startswith('claude'):
        return f'anthropic/{raw}'

    return raw


def resolve_api_key(model_name: str, explicit_api_key: str = '') -> str:
    """Pick the matching provider API key based on model name."""
    if explicit_api_key:
        return explicit_api_key

    normalized = (model_name or '').lower()
    if normalized.startswith('gemini'):
        return os.environ.get('GEMINI_API_KEY', '')
    if normalized.startswith('claude'):
        return os.environ.get('ANTHROPIC_API_KEY', '')
    if normalized.startswith('openai') or normalized.startswith('gpt'):
        return os.environ.get('OPENAI_API_KEY', '')

    # Fallback: prefer OPENAI, then GEMINI
    return os.environ.get('OPENAI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')

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
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{image_base64}'
