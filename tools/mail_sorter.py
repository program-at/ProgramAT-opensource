"""
Mail Sorter Tool for Blind Users

Helps blind users sort through physical mail by deciding whether each item
is worth opening or can be ignored.

One-shot mode: Uses Google Cloud Vision OCR to extract envelope text, then
Gemini to classify the mail and identify sender/recipient.

Streaming mode: Handled by Gemini Live (custom_gpt=True,
gpt_query="Should I open this mail?"). The model answers every ~5 seconds
with a fresh camera frame, keeping responses to ≤15 words.

Output format:
- "Ignore" — advertisement, promotional offer, coupon, or junk mail.
- "Open. From [sender] for [recipient]." — important mail; includes sender
  and recipient names when visible.
- "No mail found. [Brief scene description]." — no mail-like item visible.

In streaming mode responses are automatically capped at ~15 words by the
Gemini Live brevity prefix in the backend.
"""

import cv2
import numpy as np
import os
import json
import base64
import io
import re
from typing import Dict, List, Optional, Any

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

try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False

# Cached Vision client (avoids re-creating per frame)
_vision_client = None
_vision_client_key = None


# ---------------------------------------------------------------------------
# Image helpers (building blocks shared with other tools)
# ---------------------------------------------------------------------------

def _resize_image(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image so the longest dimension is at most max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# OCR via Google Cloud Vision
# ---------------------------------------------------------------------------

def _get_vision_client(credentials_path: str):
    """Return a cached Vision API client."""
    global _vision_client, _vision_client_key
    if _vision_client and _vision_client_key == credentials_path:
        return _vision_client

    from google.oauth2 import service_account

    if os.path.isfile(credentials_path):
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        client = vision.ImageAnnotatorClient(credentials=creds)
    elif credentials_path.startswith('{'):
        creds = service_account.Credentials.from_service_account_info(
            json.loads(credentials_path)
        )
        client = vision.ImageAnnotatorClient(credentials=creds)
    else:
        client = vision.ImageAnnotatorClient()

    _vision_client = client
    _vision_client_key = credentials_path
    return client


def extract_text_from_image(image: np.ndarray) -> str:
    """
    Run Google Cloud Vision OCR on the image and return all detected text.
    Falls back to an empty string when the API is unavailable.
    """
    if not VISION_API_AVAILABLE:
        return ""

    credentials_path = (
        os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        or os.environ.get('GOOGLE_CLOUD_VISION_API_KEY', '')
        or os.environ.get('GOOGLE_API_KEY', '')
    )
    if not credentials_path:
        return ""

    try:
        client = _get_vision_client(credentials_path)
        success, encoded = cv2.imencode('.jpg', image)
        if not success:
            return ""

        vision_image = vision.Image(content=encoded.tobytes())
        response = client.text_detection(
            image=vision_image,
            image_context=vision.ImageContext(language_hints=['en'])
        )
        if response.error.message:
            return ""

        annotations = response.text_annotations
        if annotations:
            return annotations[0].description.strip()
        return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Gemini classification
# ---------------------------------------------------------------------------

_MAIL_CRITERIA = """Classification rules:
- IGNORE: advertisements, promotional offers, coupons, catalogs, political flyers,
  charity solicitations, or any obvious junk mail.
- OPEN: bank statements, bills, legal notices, government mail, medical correspondence,
  personal letters, cheques, packages, or anything that could require action.
- NO MAIL: no mail item is visible in the image."""

_JUNK_SIGNALS = (
    'junk', 'advertisement', 'advertising', 'promo', 'promotion',
    'offer', 'coupon', 'catalog', 'flyer', 'solicitation', 'spam'
)
_IMPORTANT_SIGNALS = (
    'bank', 'statement', 'bill', 'invoice', 'government', 'legal',
    'medical', 'letter', 'personal', 'package', 'parcel', 'check', 'cheque'
)
_NO_MAIL_SIGNALS = ('no mail', 'no envelope', 'no letter', 'no package visible')
_STREAMING_WORD_LIMIT = 15


_CLASSIFICATION_PROMPT = """You are helping a blind person sort their physical mail.

You will be given:
1. An image of a mail item (envelope, package, postcard, etc.)
2. Any text detected on the mail by OCR (may be empty if OCR failed)

Your task: decide whether this is junk mail, not-junk mail that should be opened, or no-mail-visible.
You MUST make the decision for the user.

{mail_criteria}

Response format (follow exactly):
- If IGNORE → reply with exactly: Ignore
- If OPEN → reply with exactly: Open. From [sender name or company] for [recipient name].
  Use "sender not visible" / "recipient not visible" when a name is not legible.
- If NO MAIL → reply with exactly: No mail found. [One sentence describing the scene.]

OCR text detected on this image:
{ocr_text}

Keep the response under 20 words total. Do not add any extra explanation."""


def classify_mail(
    image: np.ndarray,
    ocr_text: str,
    api_key: Optional[str] = None,
    model_name: str = 'gemini-3-flash-preview',
) -> str:
    """
    Use Gemini to classify mail as Ignore / Open / No Mail Found.
    Returns a plain string ready for TTS.
    """
    if not LITELLM_AVAILABLE:
        return "Classification unavailable: LiteLLM not installed."

    api_key = resolve_api_key(model_name, api_key or '')
    if not api_key:
        return "Classification unavailable: API key not configured."

    try:
        processed = _resize_image(image, max_dim=1024)
        pil_img = _bgr_to_pil(processed)
        image_uri = pil_image_to_data_uri(pil_img)

        prompt = _CLASSIFICATION_PROMPT.format(
            mail_criteria=_MAIL_CRITERIA,
            ocr_text=ocr_text if ocr_text else "(none detected)"
        )

        full_model = resolve_model_name(model_name)
        response = litellm.completion(
            model=full_model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_uri}},
                    ],
                }
            ],
            api_key=api_key,
        )
        return _normalize_mail_decision(extract_text(response).strip(), streaming=False)
    except Exception as e:
        return f"Error classifying mail: {str(e)}"


# ---------------------------------------------------------------------------
# Streaming-mode classification (≤15 words)
# ---------------------------------------------------------------------------

_STREAMING_PROMPT = """You are helping a blind person sort mail in real time.

Look at the current camera frame and answer: should this mail be opened?

{mail_criteria}

Response format:
- If IGNORE → say: Ignore
- If OPEN → say: Open. From [sender] for [recipient].
- If NO MAIL → say: No mail found. [brief scene, ≤5 words]

You MUST decide for the user. Never say "up to you", "unclear", "maybe", or ask questions.

Keep the ENTIRE response under 15 words. No extra words.

OCR text on this image:
{ocr_text}"""


def _normalize_mail_decision(raw_text: str, streaming: bool = False) -> str:
    """
    Enforce deterministic tool output:
    - Ignore
    - Open. From ... for ...
    - No mail found. ...
    """
    cleaned = ' '.join((raw_text or '').split())
    if not cleaned:
        return "Ignore"

    lower = cleaned.lower()

    if lower.startswith('no mail found'):
        normalized = cleaned
    elif lower.startswith('ignore'):
        normalized = "Ignore"
    elif lower.startswith('open'):
        normalized = cleaned
    else:
        if any(signal in lower for signal in _NO_MAIL_SIGNALS):
            normalized = "No mail found."
        elif any(signal in lower for signal in _IMPORTANT_SIGNALS):
            normalized = "Open. From sender not visible for recipient not visible."
        elif any(signal in lower for signal in _JUNK_SIGNALS):
            normalized = "Ignore"
        else:
            # Conservative default: assume junk mail unless there is a clear important-mail signal.
            normalized = "Ignore"

    # Ensure Open output always includes sender and recipient placeholders.
    if normalized.casefold().startswith('open'):
        word_tokens = set(re.findall(r"\b\w+\b", normalized.casefold()))
        if 'from' not in word_tokens or 'for' not in word_tokens:
            normalized = "Open. From sender not visible for recipient not visible."

    if streaming:
        words = normalized.split()
        if len(words) > _STREAMING_WORD_LIMIT:
            normalized = ' '.join(words[:_STREAMING_WORD_LIMIT])
            if not normalized.endswith(('.', '!', '?')):
                normalized = normalized.rstrip('.,;:') + '.'

    return normalized


def classify_mail_streaming(
    image: np.ndarray,
    ocr_text: str,
    api_key: Optional[str] = None,
    model_name: str = 'gemini-3-flash-preview',
) -> str:
    """
    Streaming-mode classification — enforces ≤15 word output.
    Returns "" when nothing useful can be said (keeps the audio silent).
    """
    if not LITELLM_AVAILABLE:
        return ""

    api_key = resolve_api_key(model_name, api_key or '')
    if not api_key:
        return ""

    try:
        processed = _resize_image(image, max_dim=640)
        pil_img = _bgr_to_pil(processed)
        image_uri = pil_image_to_data_uri(pil_img)

        prompt = _STREAMING_PROMPT.format(
            mail_criteria=_MAIL_CRITERIA,
            ocr_text=ocr_text if ocr_text else "(none)"
        )

        full_model = resolve_model_name(model_name)
        response = litellm.completion(
            model=full_model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_uri}},
                    ],
                }
            ],
            api_key=api_key,
        )
        return _normalize_mail_decision(extract_text(response).strip(), streaming=True)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main entry point — called in one-shot mode (single captured frame).

    In streaming / Gemini Live mode this function is not invoked; the backend
    sends each frame directly to Gemini Live using the gpt_query
    "Should I open this mail?" defined on the linked issue.

    Args:
        image: Camera frame (BGR numpy array from OpenCV). May be None.
        input_data: Optional config dict:
            - 'mode': 'stream' for brief output, otherwise one-shot (default)
            - 'api_key': Gemini API key override
            - 'model': Gemini model name override (default: gemini-3-flash-preview)

    Returns:
        dict with 'audio' and 'text' keys, or plain str for simple cases.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {'type': 'error', 'text': 'No camera image available.', 'rate': 1.0, 'interrupt': False},
            'text': 'No camera image available.',
        }

    if input_data is None:
        input_data = {}

    mode = input_data.get('mode', 'oneshot')
    api_key = input_data.get('api_key')
    model = input_data.get(
        'model',
        os.environ.get('LLM_MODEL', os.environ.get('GEMINI_MODEL', 'gemini-3-flash-preview'))
    )

    # Step 1: OCR — extract any text visible on the mail item
    ocr_text = extract_text_from_image(image)

    # Step 2: Classify using Gemini
    if mode == 'stream':
        result_text = classify_mail_streaming(image, ocr_text, api_key=api_key, model_name=model)
        if not result_text:
            return ""  # silent frame — nothing changed
    else:
        result_text = classify_mail(image, ocr_text, api_key=api_key, model_name=model)

    if not result_text:
        return ""

    # Step 3: Choose audio type based on classification
    lower = result_text.lower()
    if lower.startswith('open'):
        audio_type = 'success'
    elif lower.startswith('ignore'):
        audio_type = 'speech'
    else:
        audio_type = 'speech'

    return {
        'audio': {
            'type': audio_type,
            'text': result_text,
            'rate': 1.0,
            'interrupt': False,
        },
        'text': result_text,
    }
