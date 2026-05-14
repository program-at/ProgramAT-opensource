"""
Plug Point Locator for Blind Users

A tool that helps blind users locate plug points (electrical outlets) in unfamiliar
spaces using camera input and audio feedback.

Features:
- Detects plug points / electrical outlets using YoloWorld (not in COCO classes)
- Announces "no plug point" when none is detected
- Provides clockface directions (1-3 and 9-12 o'clock) to detected plug points
- Audio-optimized output for text-to-speech
- Streaming mode limited to 15 words

Navigation System:
- Uses clock face positions (1-3 o'clock on right, 9-12 o'clock on left)
- Positions 4-8 not used as they would be behind the camera

Example Usage:
User moves camera around an unfamiliar space. The tool says "no plug point" until
one is detected, at which point it gives clockface directions, e.g.
"Plug point at 3 o'clock"

Model caching is handled by the backend server via a shared `yolo_model_cache`
dictionary that persists across all tool executions.

This tool runs on the backend server and receives camera frames from the mobile app.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Detection constants
DEFAULT_CONFIDENCE = 0.25

PLUG_CLASSES = [
    'electrical outlet', 'power outlet', 'plug socket', 'wall socket',
    'power socket', 'plug point', 'electrical socket', 'outlet',
    'power plug', 'wall plug', 'electrical plug', 'socket',
    'mains socket', 'power receptacle', 'electrical receptacle',
]

# Only clock positions visible from camera (not behind it)
VALID_CLOCK_POSITIONS = [12, 1, 2, 3, 9, 10, 11]

# Streaming mode word limit (per instructions)
STREAMING_WORD_LIMIT = 15

# Temporal smoothing: require this many consecutive no-detection frames before
# announcing "no plug point" to avoid noisy flickering output
NO_PLUG_ANNOUNCEMENT_DELAY = 5

# Global state for streaming mode
_last_plug_state = None  # None, 'no_plug', or a clock-position string
_consecutive_no_plug_frames = 0


def detect_plug_points(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Detect plug points / electrical outlets using YoloWorld.

    Since 'plug point' and 'electrical outlet' are not in the standard COCO
    dataset, YoloWorld is used for open-vocabulary detection.

    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)

    Returns:
        List of detection dictionaries with keys:
            - class_name: Human-readable class name
            - confidence: Detection confidence (0.0 to 1.0)
            - bbox: Bounding box [x, y, width, height]
            - center: Center point [x, y]
    """
    if image is None or image.size == 0:
        return []

    detections = []

    try:
        from ultralytics import YOLO

        # Shared model cache injected by the backend server
        model_cache = globals().get('yolo_model_cache', {})
        cache_key = 'plug_point_locator_yolov8s-world'

        if cache_key not in model_cache:
            model_cache[cache_key] = {
                'model': YOLO('yolov8s-world.pt'),
                'classes': None
            }

        model_info = model_cache[cache_key]
        yolo_model = model_info['model']

        # Set custom classes when first loaded or if they changed
        classes_key = tuple(PLUG_CLASSES)
        if model_info['classes'] is None or model_info['classes'] != classes_key:
            yolo_model.set_classes(PLUG_CLASSES)
            model_info['classes'] = classes_key

        results = yolo_model(image, conf=confidence_threshold, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = PLUG_CLASSES[class_id] if 0 <= class_id < len(PLUG_CLASSES) else "plug point"

                center_x = x + w // 2
                center_y = y + h // 2

                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'center': [center_x, center_y]
                })

    except ImportError:
        print("Warning: ultralytics not available for plug point detection")
    except Exception as e:
        print(f"Warning: Plug point detection error: {e}")

    return detections


def get_clock_position(center_x: int, center_y: int, frame_width: int, frame_height: int) -> int:
    """
    Determine clock face position of an object.

    Only returns positions visible from camera perspective:
    - 12: Top center (and center of frame)
    - 1-3: Right side (top to bottom)
    - 9-11: Left side (top to bottom)

    Args:
        center_x: X coordinate of object center
        center_y: Y coordinate of object center
        frame_width: Width of the frame
        frame_height: Height of the frame

    Returns:
        Clock position (1-3 or 9-12)
    """
    norm_x = center_x / frame_width
    norm_y = center_y / frame_height

    # Horizontal region
    if norm_x < 0.33:
        h_region = 'left'
    elif norm_x < 0.67:
        h_region = 'center'
    else:
        h_region = 'right'

    # Vertical region
    if norm_y < 0.33:
        v_region = 'top'
    elif norm_y < 0.67:
        v_region = 'middle'
    else:
        v_region = 'bottom'

    # Map to clock positions
    if h_region == 'center' and v_region == 'top':
        return 12

    if h_region == 'right':
        if v_region == 'top':
            return 1
        elif v_region == 'middle':
            return 2
        else:
            return 3

    if h_region == 'left':
        if v_region == 'top':
            return 11
        elif v_region == 'middle':
            return 10
        else:
            return 9

    # Center (not top) → treat as straight ahead
    return 12


def _clock_text(clock_pos: int) -> str:
    """Return natural language text for a clock position."""
    if clock_pos == 12:
        return "straight ahead"
    return f"at {clock_pos} o'clock"


def _truncate_to_word_limit(text: str, limit: int = STREAMING_WORD_LIMIT) -> str:
    words = text.split()
    if len(words) > limit:
        return " ".join(words[:limit])
    return text


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """
    Main entry point for the plug point locator tool.

    Detects plug points / electrical outlets and provides verbal clockface
    navigation instructions for blind users.

    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration dict:
            - confidence: Detection threshold (default 0.25)
            - is_streaming: Streaming mode flag (default False)

    Returns:
        Audio-friendly string or dict with 'audio'/'text' keys.
        In streaming mode returns "" when the scene has not changed.
    """
    global _last_plug_state, _consecutive_no_plug_frames

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {'type': 'error', 'text': 'No camera image available', 'rate': 1.0},
            'text': 'No camera image available'
        }

    config = input_data if isinstance(input_data, dict) else {}
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    is_streaming = config.get('is_streaming', False)

    height, width = image.shape[:2]

    detections = detect_plug_points(image, confidence)

    # ------------------------------------------------------------------ #
    # No plug point detected                                               #
    # ------------------------------------------------------------------ #
    if not detections:
        _consecutive_no_plug_frames += 1

        if is_streaming:
            # Announce only after several consecutive frames without detection
            if (_last_plug_state != 'no_plug' and
                    _consecutive_no_plug_frames >= NO_PLUG_ANNOUNCEMENT_DELAY):
                _last_plug_state = 'no_plug'
                return {
                    'audio': {'type': 'speech', 'text': 'No plug point', 'rate': 1.0},
                    'text': 'No plug point'
                }
            return ""
        else:
            # One-shot mode: always respond
            _last_plug_state = None
            return {
                'audio': {'type': 'speech', 'text': 'No plug point detected', 'rate': 1.0},
                'text': 'No plug point detected'
            }

    # ------------------------------------------------------------------ #
    # Plug point(s) detected                                               #
    # ------------------------------------------------------------------ #
    _consecutive_no_plug_frames = 0

    # Use the largest (most prominent) detection
    main_det = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
    cx, cy = main_det['center']
    clock_pos = get_clock_position(cx, cy, width, height)

    count = len(detections)
    if count == 1:
        msg = f"Plug point {_clock_text(clock_pos)}"
    else:
        msg = f"{count} plug points, nearest {_clock_text(clock_pos)}"

    msg = _truncate_to_word_limit(msg)

    if is_streaming:
        state_key = f"plug_at_{clock_pos}"
        if _last_plug_state == state_key:
            # No change – stay silent to avoid repetition
            return ""
        _last_plug_state = state_key
        return {
            'audio': {'type': 'speech', 'text': msg, 'rate': 1.0},
            'text': msg
        }
    else:
        # One-shot mode: always respond
        _last_plug_state = None
        return {
            'audio': {'type': 'speech', 'text': msg, 'rate': 1.0},
            'text': msg
        }
