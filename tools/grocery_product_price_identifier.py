import os
import re
import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


DEFAULT_CONFIDENCE = 0.25
DEFAULT_TRACK_MODE = True
MAX_PRODUCT_NAME_WORDS = 6
MAX_STREAMING_WORDS = 15
MAX_WHOLE_DOLLAR_PRICE = 100.0
YOLO_MODEL_CACHE_KEY = 'yolo11n'
STATE_CACHE_KEY = 'grocery_product_price_state'
LOGGER = logging.getLogger(__name__)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

STORE_ITEM_CLASSES = {
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'sports ball',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'potted plant', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
}

try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False


PRICE_PATTERN = re.compile(
    r"(?:\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)|(?<!\d)(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})(?!\d))"
)


def _get_shared_cache() -> Dict[str, Any]:
    return globals().get('yolo_model_cache', {})


def _get_or_load_yolo_model() -> Any:
    from ultralytics import YOLO

    cache = _get_shared_cache()
    if YOLO_MODEL_CACHE_KEY not in cache:
        LOGGER.info("Loading YOLO model for store-item detection")
        cache[YOLO_MODEL_CACHE_KEY] = YOLO('yolo11n.pt')
    else:
        LOGGER.debug("Reusing cached YOLO model for store-item detection")
    return cache[YOLO_MODEL_CACHE_KEY]


def detect_products(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    if image is None or image.size == 0:
        LOGGER.warning("detect_products called with empty image")
        return []

    try:
        model = _get_or_load_yolo_model()
        results = model(image, conf=confidence_threshold, verbose=False)
    except Exception:
        LOGGER.exception("YOLO store-item detection failed")
        return []

    detections: List[Dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"object_{class_id}"
            if class_name not in STORE_ITEM_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            detections.append({
                'class_name': class_name,
                'confidence': float(box.conf[0]),
                'bbox': [x, y, w, h],
                'center': [x + w // 2, y + h // 2],
            })
    LOGGER.info("Detected %d store-item candidates", len(detections))
    return detections


def _focus_score(detection: Dict[str, Any], frame_w: int, frame_h: int) -> float:
    x, y, w, h = detection['bbox']
    area_score = max(0.0, min(1.0, (w * h) / float(max(1, frame_w * frame_h))))

    cx, cy = detection['center']
    dx = cx - frame_w / 2.0
    dy = cy - frame_h / 2.0
    dist = (dx * dx + dy * dy) ** 0.5
    max_dist = ((frame_w / 2.0) ** 2 + (frame_h / 2.0) ** 2) ** 0.5
    center_score = 1.0 - min(1.0, dist / max(1e-6, max_dist))

    return 0.65 * area_score + 0.35 * center_score


def get_focus_product(detections: List[Dict[str, Any]], frame_w: int, frame_h: int) -> Optional[Dict[str, Any]]:
    if not detections:
        return None
    return max(detections, key=lambda d: _focus_score(d, frame_w, frame_h))


def crop_focus_region(image: np.ndarray, bbox: List[int], padding_ratio: float = 0.15) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    return image[y1:y2, x1:x2]


def _vision_client(api_key: Optional[str] = None) -> Optional[Any]:
    if not VISION_API_AVAILABLE:
        LOGGER.info("Google Vision not available in environment")
        return None

    if api_key is None:
        api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')

    try:
        from google.oauth2 import service_account
        import json

        if api_key and os.path.isfile(api_key):
            LOGGER.debug("Creating Google Vision client from service-account file")
            credentials = service_account.Credentials.from_service_account_file(api_key)
            return vision.ImageAnnotatorClient(credentials=credentials)
        if api_key and api_key.startswith('{'):
            LOGGER.debug("Creating Google Vision client from service-account JSON string")
            credentials_dict = json.loads(api_key)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return vision.ImageAnnotatorClient(credentials=credentials)
        LOGGER.debug("Creating default Google Vision client")
        return vision.ImageAnnotatorClient()
    except Exception:
        LOGGER.exception("Failed to initialize Google Vision client")
        return None


def _extract_text_google_vision(image: np.ndarray, api_key: Optional[str] = None, language: str = 'en') -> str:
    client = _vision_client(api_key)
    if client is None:
        LOGGER.info("Skipping Google Vision OCR because no client is available")
        return ''

    try:
        success, encoded = cv2.imencode('.jpg', image)
        if not success:
            LOGGER.warning("Failed to encode focus region for Google Vision OCR")
            return ''

        vision_image = vision.Image(content=encoded.tobytes())
        context = vision.ImageContext(language_hints=[language])
        response = client.text_detection(image=vision_image, image_context=context)
        error_message = getattr(getattr(response, 'error', None), 'message', '')
        if error_message or not response.text_annotations:
            LOGGER.info("Google Vision OCR returned no text for focus region")
            return ''
        LOGGER.info("Google Vision OCR succeeded for focus region")
        return response.text_annotations[0].description.strip()
    except Exception:
        LOGGER.exception("Google Vision OCR failed")
        return ''


def _extract_text_tesseract(image: np.ndarray) -> str:
    try:
        import pytesseract
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        text = pytesseract.image_to_string(rgb)
        LOGGER.info("Tesseract OCR completed for focus region")
        return text.strip()
    except Exception:
        LOGGER.exception("Tesseract OCR failed")
        return ''


def extract_text_from_region(region: np.ndarray, api_key: Optional[str] = None, language: str = 'en') -> str:
    text = _extract_text_google_vision(region, api_key=api_key, language=language)
    if text:
        return text
    LOGGER.info("Falling back to Tesseract OCR for focus region")
    return _extract_text_tesseract(region)


def extract_price(text: str) -> Optional[str]:
    if not text:
        return None
    match = PRICE_PATTERN.search(text)
    if not match:
        return None

    amount = (match.group(1) or match.group(2) or '').replace(',', '')
    if not amount:
        return None
    if '.' not in amount:
        if float(amount) > MAX_WHOLE_DOLLAR_PRICE:
            return None
        amount = f"{amount}.00"
    return f"${amount}"


def extract_product_name(text: str) -> Optional[str]:
    if not text:
        return None

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = ' '.join(raw_line.strip().split())
        if not line:
            continue
        if PRICE_PATTERN.search(line):
            line = PRICE_PATTERN.sub('', line).strip(' -:')
        if len(line) < 2:
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return None

    best = max(cleaned_lines, key=lambda s: len(s))
    words = best.split()
    return ' '.join(words[:MAX_PRODUCT_NAME_WORDS])


def parse_product_and_price(ocr_text: str, fallback_class: str) -> Dict[str, Optional[str]]:
    price = extract_price(ocr_text)
    name = extract_product_name(ocr_text)
    if not name:
        name = fallback_class.replace('_', ' ')
    return {'name': name, 'price': price}


def _build_message(name: str, price: Optional[str], track_mode: bool) -> str:
    if price:
        message = f"{name}, price {price}."
    else:
        message = f"{name}. Price not visible."

    if not track_mode:
        return message

    words = message.split()
    if len(words) <= MAX_STREAMING_WORDS:
        return message
    return ' '.join(words[:MAX_STREAMING_WORDS])


def _state_cache() -> Dict[str, Any]:
    cache = _get_shared_cache()
    if STATE_CACHE_KEY not in cache:
        cache[STATE_CACHE_KEY] = {'last_message': ''}
    return cache[STATE_CACHE_KEY]


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        LOGGER.warning("grocery_product_price_identifier.main received invalid image")
        return "No camera image available"

    config = input_data if isinstance(input_data, dict) else {}
    confidence = float(config.get('confidence', DEFAULT_CONFIDENCE))
    track_mode = bool(config.get('track_mode', DEFAULT_TRACK_MODE))
    language = config.get('language', 'en')
    api_key = config.get('api_key')
    LOGGER.info("Starting store-item product/price processing (track_mode=%s, confidence=%.2f)", track_mode, confidence)

    detections = detect_products(image, confidence_threshold=confidence)
    if not detections:
        LOGGER.info("No store-item detections found in frame")
        if track_mode:
            return ""
        return "No store item in focus."

    frame_h, frame_w = image.shape[:2]
    focus = get_focus_product(detections, frame_w, frame_h)
    if not focus:
        LOGGER.info("No focus product selected from detections")
        if track_mode:
            return ""
        return "No store item in focus."
    LOGGER.info("Selected focus product class=%s confidence=%.2f", focus['class_name'], focus.get('confidence', 0.0))

    region = crop_focus_region(image, focus['bbox'])
    LOGGER.info("Cropped focus region for OCR (w=%d, h=%d)", region.shape[1], region.shape[0])
    ocr_text = extract_text_from_region(region, api_key=api_key, language=language)
    parsed = parse_product_and_price(ocr_text, focus['class_name'])
    LOGGER.info("Parsed product result name=%s price=%s", parsed.get('name'), parsed.get('price'))

    message = _build_message(parsed['name'] or focus['class_name'], parsed['price'], track_mode)

    state = _state_cache()
    if track_mode and message == state.get('last_message', ''):
        LOGGER.debug("Suppressing repeated announcement in track mode")
        return ""
    state['last_message'] = message
    LOGGER.info("Returning grocery product-price response")

    return {
        'audio': {
            'type': 'speech',
            'text': message,
            'rate': 1.0,
            'interrupt': bool(track_mode)
        },
        'text': message
    }


__all__ = [
    'main',
    'detect_products',
    'get_focus_product',
    'crop_focus_region',
    'extract_text_from_region',
    'extract_price',
    'extract_product_name',
    'parse_product_and_price'
]
