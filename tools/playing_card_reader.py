"""
Playing Card Reader Tool

AI-powered playing card recognition tool that identifies playing cards held in hand
and announces their names aloud for blind or low vision users.

Features:
- Recognizes standard 52-card deck plus jokers
- Reads multiple cards from left to right
- Uses Google Gemini Vision API for accurate card recognition
- Streaming mode output capped at 15 words
- Audio-optimized output for text-to-speech

Audio Output:
- Returns natural language card names suitable for text-to-speech
- Example: "King of hearts, 5 of spades, ace of diamonds."
"""

import cv2
import numpy as np
from typing import Dict, Optional, Any
import os
import base64
import io
from PIL import Image
from litellm_utils import (
    resolve_model_name,
    resolve_api_key,
    extract_text,
    pil_image_to_data_uri,
)

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False


def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    """Resize image efficiently while maintaining aspect ratio."""
    height, width = image.shape[:2]
    max_width, max_height = max_size
    if width <= max_width and height <= max_height:
        return image
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB format."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def build_card_prompt(streaming: bool = False) -> str:
    """Build the prompt for playing card recognition."""
    base = (
        "You are a playing card reader for a blind user. "
        "Look at this image and identify any playing cards visible. "
        "A standard deck has 52 cards (ace through king in hearts, diamonds, clubs, spades) plus jokers. "
    )

    if streaming:
        instruction = (
            "List only the card names you see, ordered from left to right. "
            "Use names like 'ace of spades', 'king of hearts', '5 of diamonds', 'joker'. "
            "IMPORTANT: Keep your entire response under 15 words. "
            "If no cards are visible, reply with exactly: no cards visible. "
            "Examples: 'King of hearts, 5 of spades.' "
            "or: 'Ace of clubs, queen of diamonds, 3 of hearts.'"
        )
    else:
        instruction = (
            "List all the card names you see, ordered from left to right. "
            "Use names like 'ace of spades', 'king of hearts', '5 of diamonds', 'joker'. "
            "If no cards are visible, say: no cards visible. "
            "Be concise — just list the card names separated by commas, ending with a period. "
            "Examples: 'King of hearts, 5 of spades, ace of diamonds.' "
            "or: 'Ace of clubs, queen of diamonds, 3 of hearts, 7 of spades.'"
        )

    return base + instruction


def identify_cards(
    image: np.ndarray,
    api_key: Optional[str] = None,
    streaming: bool = False,
    model_name: str = 'gemini-3-flash-preview'
) -> Dict[str, Any]:
    """
    Core function that performs AI-powered playing card identification using Gemini.

    Args:
        image: OpenCV image (numpy array in BGR format)
        api_key: Gemini API key (uses env var if not provided)
        streaming: Whether to apply 15-word streaming limit
        model_name: Gemini model to use

    Returns:
        Dictionary with results:
        {
            'success': bool,
            'description': str,
            'cards_found': bool
        }
    """
    if not LITELLM_AVAILABLE:
        return {
            'success': False,
            'description': 'LiteLLM not available. Please install litellm package.',
            'cards_found': False,
        }

    api_key = resolve_api_key(model_name, api_key)

    if not api_key:
        return {
            'success': False,
            'description': 'API key not configured. Please set GEMINI_API_KEY in the environment.',
            'cards_found': False,
        }

    try:
        processed_image = resize_image_if_needed(image, max_size=(1024, 1024))
        pil_image = convert_cv2_to_pil(processed_image)
        image_data_uri = pil_image_to_data_uri(pil_image)

        prompt = build_card_prompt(streaming=streaming)
        model_name = resolve_model_name(model_name)

        response = litellm.completion(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_data_uri}},
                    ],
                }
            ],
            api_key=api_key,
        )

        description = extract_text(response)
        cards_found = 'no cards' not in description.lower()

        return {
            'success': True,
            'description': description,
            'cards_found': cards_found,
        }

    except Exception as e:
        return {
            'success': False,
            'description': f'Card recognition failed: {str(e)}',
            'cards_found': False,
        }


def format_for_audio(description: str, max_words: Optional[int] = None) -> str:
    """Format card description for audio output."""
    formatted = ' '.join(description.strip().split())

    if max_words is not None:
        words = formatted.split()
        if len(words) > max_words:
            formatted = ' '.join(words[:max_words])
            if not formatted.endswith(('.', '!', '?')):
                formatted = formatted.rstrip('.,;:') + '.'

    return formatted


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Main entry point for the playing card reader tool.

    Identifies playing cards visible in the camera frame and announces
    their names from left to right.

    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration dictionary:
            - 'streaming': bool, apply 15-word limit (default: False)
            - 'api_key': Optional API key override
            - 'model': Optional Gemini model override

    Returns:
        Audio-friendly string or dict with 'audio'/'text' keys.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {
                'type': 'error',
                'text': 'No camera image available for card recognition.',
                'rate': 1.0,
                'interrupt': False,
            },
            'text': 'No image available.',
        }

    if input_data is None:
        input_data = {}

    streaming = bool(input_data.get('streaming', False))
    api_key = input_data.get('api_key')
    model = input_data.get(
        'model',
        os.environ.get('LLM_MODEL', os.environ.get('GEMINI_MODEL', 'gemini-3-flash-preview')),
    )

    result = identify_cards(
        image=image,
        api_key=api_key,
        streaming=streaming,
        model_name=model,
    )

    if not result['success']:
        return {
            'audio': {
                'type': 'error',
                'text': result['description'],
                'rate': 1.0,
                'interrupt': False,
            },
            'text': result['description'],
        }

    max_words = 15 if streaming else None
    description = format_for_audio(result['description'], max_words=max_words)

    if not result['cards_found']:
        return 'No cards visible.'

    return description
