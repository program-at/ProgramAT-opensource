"""
Shared LiteLLM helper utilities for backend modules.
"""

import base64
import io
import logging
import os
from PIL import Image


logger = logging.getLogger(__name__)


class TaskType:
    TEMPLATE_FILL = 'TEMPLATE_FILL'
    TOOL_MATCHING = 'TOOL_MATCHING'
    PROGRAM_GENERATION = 'PROGRAM_GENERATION'
    OCR = 'OCR'
    IMAGE_ANALYSIS = 'IMAGE_ANALYSIS'
    LIVE_CAMERA_ANALYSIS = 'LIVE_CAMERA_ANALYSIS'
    PLANNING = 'PLANNING'
    CODE_GENERATION = 'CODE_GENERATION'
    OBJECT_DETECTION = 'OBJECT_DETECTION'
    SUMMARIZATION = 'SUMMARIZATION'


class ModelTier:
    ULTRA_FAST = 'ULTRA_FAST'
    FAST = 'FAST'
    BALANCED = 'BALANCED'
    HIGH_QUALITY = 'HIGH_QUALITY'


_TASK_TO_TIER = {
    TaskType.OBJECT_DETECTION: ModelTier.ULTRA_FAST,
    TaskType.TEMPLATE_FILL: ModelTier.FAST,
    TaskType.TOOL_MATCHING: ModelTier.FAST,
    TaskType.OCR: ModelTier.FAST,
    TaskType.LIVE_CAMERA_ANALYSIS: ModelTier.FAST,
    TaskType.SUMMARIZATION: ModelTier.FAST,
    TaskType.PROGRAM_GENERATION: ModelTier.BALANCED,
    TaskType.IMAGE_ANALYSIS: ModelTier.BALANCED,
    TaskType.PLANNING: ModelTier.HIGH_QUALITY,
    TaskType.CODE_GENERATION: ModelTier.HIGH_QUALITY,
}


def _is_non_llm_model(model_name: str) -> bool:
    normalized = (model_name or '').strip().lower()
    return normalized.startswith('yolo') or normalized.endswith('.pt')


def _tier_default_model(tier: str) -> str:
    if tier == ModelTier.ULTRA_FAST:
        return os.environ.get('ULTRA_FAST_MODEL', 'yolo11n.pt')
    if tier == ModelTier.FAST:
        return os.environ.get('FAST_MODEL', 'groq/llama-3.3-70b-versatile')
    if tier == ModelTier.BALANCED:
        return os.environ.get('BALANCED_MODEL', 'claude-3-5-sonnet-20241022')
    if tier == ModelTier.HIGH_QUALITY:
        return os.environ.get('HIGH_QUALITY_MODEL', 'gpt-4.1')
    return os.environ.get('FAST_MODEL', 'groq/llama-3.3-70b-versatile')


def get_task_tier(task_type: str) -> str:
    return _TASK_TO_TIER.get((task_type or '').strip().upper(), ModelTier.FAST)


def get_model_for_task(task_type: str, requested_model: str = '') -> str:
    """Return the model to use for a task and log the routing decision."""
    tier = get_task_tier(task_type)
    override = (requested_model or '').strip()

    if override:
        if tier == ModelTier.ULTRA_FAST and not _is_non_llm_model(override):
            model_name = _tier_default_model(tier)
        else:
            model_name = override
    else:
        model_name = _tier_default_model(tier)

    if _is_non_llm_model(model_name):
        routed_model = model_name
    else:
        routed_model = resolve_model_name(model_name, default_model=model_name)

    logger.info(f"[ModelRouter] Task={task_type} -> {tier} -> {model_name}")
    return routed_model


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
    if normalized.startswith('groq'):
        return os.environ.get('GROQ_API_KEY', '')
    if normalized.startswith('claude'):
        return os.environ.get('ANTHROPIC_API_KEY', '')
    if normalized.startswith('openai') or normalized.startswith('gpt'):
        return os.environ.get('OPENAI_API_KEY', '')

    # Fallback: prefer GROQ for FAST, then OPENAI, then GEMINI
    return os.environ.get('GROQ_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')

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
