"""
Grocery Store Item and Price Identifier

Helps blind and low vision users navigate grocery stores by identifying
items and their prices from camera frames, and confirming which item
has been grabbed when the user picks one up.

Features:
- Identifies grocery products and prices visible on shelves
- Announces new items as the user scans the shelf, avoiding repetition
- Detects when an item is grabbed (fills most of the frame) and announces it
- Returns audio-friendly output for text-to-speech
- Streaming mode limited to 15 words per response

Audio Output:
- Shelf scanning: "Cheerios, 3.50" or "Lucky Charms, 4 dollars"
- Grab detection: "You grabbed Cheerios, 3.50"
- Returns "" when nothing new is detected (streaming mode)

Configuration Options (via input_data):
- model: Gemini model name (default: gemini-3-flash-preview)
- api_key: Optional Gemini API key override
- skip_frames: Frames to skip between analyses in streaming (default 5)
- reset: Reset tracking state (default False)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
import os
from PIL import Image
from collections import deque

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False

from litellm_utils import (
    resolve_model_name,
    resolve_api_key,
    extract_text,
    pil_image_to_data_uri,
)

# Constants
DEFAULT_MODEL = 'gemini-3-flash-preview'
DEFAULT_SKIP_FRAMES = 5   # Process every Nth frame in streaming mode
SIMILARITY_THRESHOLD = 0.65  # Word-level Jaccard similarity for repeat suppression
MAX_HISTORY = 6           # Number of recent announcements to track
MAX_IMAGE_SIZE = (1024, 1024)
_CACHE_KEY = 'grocery_store_state'


def _get_state() -> Dict[str, Any]:
    """
    Return the persistent state dict from yolo_model_cache.

    The backend injects yolo_model_cache as a mutable dict that is shared
    across all streaming frame executions, making it the correct place to
    store any state that must survive between frames (frame counter, history).
    """
    cache = globals().get('yolo_model_cache', {})
    if _CACHE_KEY not in cache:
        cache[_CACHE_KEY] = {
            'frame_counter': 0,
            'result_history': deque(maxlen=MAX_HISTORY),
            'last_result': '',
        }
    return cache[_CACHE_KEY]


def _resize_image(image: np.ndarray, max_size: tuple = MAX_IMAGE_SIZE) -> np.ndarray:
    """Resize image while maintaining aspect ratio, only if it exceeds max_size."""
    h, w = image.shape[:2]
    max_w, max_h = max_size
    if w <= max_w and h <= max_h:
        return image
    scale = min(max_w / w, max_h / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _convert_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB format."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate word-level Jaccard similarity between two strings."""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) if union else 0.0


def _is_duplicate(text: str, history: deque) -> bool:
    """Return True if text is too similar to a recent announcement."""
    for prev in history:
        if _calculate_similarity(text, prev) >= SIMILARITY_THRESHOLD:
            return True
    return False


def _build_prompt() -> str:
    """Build the Gemini prompt for grocery item and price identification."""
    return (
        "You are assisting a blind person shopping in a grocery store. "
        "Analyze this camera image and respond with ONLY one of these formats:\n\n"
        "1. If a SINGLE product fills most of the frame (user is holding it close to camera): "
        "respond 'You grabbed [Product Name], [price]'\n"
        "2. If scanning a shelf (multiple items visible in background): "
        "respond '[Most prominent product], [price]' — list at most 2 items separated by a period\n"
        "3. If no grocery items or readable prices are visible: respond with exactly ''\n\n"
        "Rules:\n"
        "- Keep total response under 15 words\n"
        "- Use natural spoken format: 'Cheerios, 3 dollars 50' or 'Cheerios, 3.50'\n"
        "- Only mention an item if you can identify both its name AND price\n"
        "- Do NOT include explanations, labels, or any extra text\n"
        "- Do NOT say 'I see' or 'I notice'; go straight to the product info"
    )


def _analyze_frame(
    image: np.ndarray,
    api_key: Optional[str],
    model_name: str,
) -> str:
    """
    Send a camera frame to Gemini and return the grocery announcement string.

    Returns an empty string on API error, missing key, or when no items are found.
    """
    if not LITELLM_AVAILABLE:
        return ""

    resolved_key = resolve_api_key(model_name, api_key or "")
    if not resolved_key:
        return ""

    try:
        processed = _resize_image(image)
        pil_img = _convert_to_pil(processed)
        image_uri = pil_image_to_data_uri(pil_img)
        prompt = _build_prompt()
        resolved_model = resolve_model_name(model_name)

        response = litellm.completion(
            model=resolved_model,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image_uri}},
                ],
            }],
            api_key=resolved_key,
            max_tokens=60,
        )

        result = extract_text(response).strip().strip("'\"")

        # Treat explicit empty / null-like responses as silence
        if result.lower() in ('', 'none', 'n/a', 'nothing', 'no items found'):
            return ""

        return result

    except Exception:
        return ""


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Main entry point for the grocery store item and price identifier.

    Announces grocery items and prices as the user scans a shelf, and
    confirms which item was grabbed when the user picks one up.

    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration dictionary:
            - model: Gemini model name (default: gemini-3-flash-preview)
            - api_key: Optional API key override
            - skip_frames: Frames to skip between API calls (default 5)
            - reset: Reset tracking state (default False)

    Returns:
        str or dict — audio-friendly announcement, or "" to stay silent.
        Grab announcements use 'success' audio type for distinct feedback.

    Examples:
        "Cheerios, 3.50"
        "Lucky Charms, 4 dollars. Raisin Bran, 2.50"
        {'audio': {'type': 'success', 'text': 'You grabbed Cheerios, 3.50', ...}, 'text': ...}
        ""  — nothing new detected this frame
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"

    config = input_data if isinstance(input_data, dict) else {}

    # Retrieve persistent state (survives across streaming frames via yolo_model_cache)
    state = _get_state()

    # Allow caller to reset tracking (e.g. when user moves to a new aisle)
    if config.get('reset', False):
        state['frame_counter'] = 0
        state['result_history'].clear()
        state['last_result'] = ''
        return "Grocery scanner reset"

    model = config.get(
        'model',
        os.environ.get('LLM_MODEL', os.environ.get('GEMINI_MODEL', DEFAULT_MODEL))
    )
    api_key = config.get('api_key')
    skip_frames = int(config.get('skip_frames', DEFAULT_SKIP_FRAMES))

    state['frame_counter'] += 1

    # Skip frames to reduce API costs in streaming mode
    if state['frame_counter'] % skip_frames != 0:
        return ""

    if not LITELLM_AVAILABLE:
        return "Grocery identifier requires the litellm package"

    result = _analyze_frame(image, api_key=api_key, model_name=model)

    if not result:
        return ""

    # Suppress repetitive announcements
    if _is_duplicate(result, state['result_history']):
        return ""

    state['result_history'].append(result)
    state['last_result'] = result

    # Distinguish grab confirmations with a success audio cue
    if result.lower().startswith('you grabbed'):
        return {
            'audio': {
                'type': 'success',
                'text': result,
                'rate': 1.0,
                'interrupt': True,
            },
            'text': result,
        }

    return result


def reset_tracking():
    """Reset all tracking state. Useful between aisles or test runs."""
    state = _get_state()
    state['frame_counter'] = 0
    state['result_history'].clear()
    state['last_result'] = ''


# Building block exports
__all__ = [
    'main',
    'reset_tracking',
]
