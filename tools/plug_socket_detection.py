"""
Plug Socket Detection Tool

Helps blind and low-vision users locate available electrical wall outlets using
camera input and directional audio feedback.

Features:
- Detects visible electrical outlets/plug sockets using YoloWorld
- Focuses on outlets where prong slots are open/uncovered
- Reports quantity and clock-face position of detected outlets (1-3 and 9-12 only)
- Identifies the closest outlet when multiple are present
- Audio-optimised output for text-to-speech
- Streaming mode limited to 15 words

Navigation System:
- Clock face positions visible from camera perspective (1-3, 9-12 o'clock)
- Positions 4-8 not used as they would be behind the camera

Example Output:
- Single outlet slightly left: "1 plug socket at 11 o'clock"
- Two outlets, nearest straight ahead: "2 plug sockets found, nearest at 12 o'clock"
- No outlets: "no plug sockets found"

Model caching is handled by the backend server via a shared `yolo_model_cache`
dictionary that persists across all tool executions, enabling real-time performance.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Detection confidence threshold
# YoloWorld can produce lower confidence for non-COCO objects
DEFAULT_CONFIDENCE = 0.20

# Classes used to prompt YoloWorld for open/available outlets
SOCKET_CLASSES = [
    'electrical outlet', 'wall outlet', 'power outlet', 'plug socket',
    'electrical socket', 'power socket', 'wall socket',
    'electrical receptacle', 'power receptacle',
    'outlet with open slots', 'available outlet', 'power plug point',
]

# Only use clock positions visible from camera (not 4-8, which are behind)
VALID_CLOCK_POSITIONS = [12, 1, 2, 3, 9, 10, 11]

# Streaming word limit
STREAMING_WORD_LIMIT = 15

# Temporal smoothing: require this many consecutive "nothing" frames before
# announcing "no plug sockets found" in streaming mode
NO_SOCKET_FRAME_THRESHOLD = 5

# Module-level globals for streaming state.
# These reset each time the module is exec()-ed for a new session, but persist
# within a streaming session because the backend reuses the same exec namespace.
_last_announced = None   # Last spoken message (for dedup in streaming)
_consecutive_no_socket_frames = 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_sockets(image: np.ndarray, confidence: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Detect electrical outlets/plug sockets in the image using YoloWorld.

    Args:
        image: BGR image as a numpy array.
        confidence: Minimum detection confidence (0.0 – 1.0).

    Returns:
        List of detection dicts, each with keys:
            class_name, confidence, bbox ([x, y, w, h]), center ([cx, cy])
    """
    if image is None or image.size == 0:
        return []

    detections = []

    try:
        from ultralytics import YOLO

        # Use the injected shared model cache to avoid reloading every frame
        model_cache = globals().get('yolo_model_cache', {})
        cache_key = 'plug_socket_yolov8s_world'

        if cache_key not in model_cache:
            model_cache[cache_key] = {
                'model': YOLO('yolov8s-world.pt'),
                'classes': None,
            }

        entry = model_cache[cache_key]
        model = entry['model']

        # Set custom classes when they change (or on first use)
        classes_tuple = tuple(SOCKET_CLASSES)
        if entry['classes'] != classes_tuple:
            model.set_classes(SOCKET_CLASSES)
            entry['classes'] = classes_tuple

        results = model(image, conf=confidence, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                class_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = (
                    SOCKET_CLASSES[class_id]
                    if 0 <= class_id < len(SOCKET_CLASSES)
                    else 'outlet'
                )
                cx = x + w // 2
                cy = y + h // 2
                detections.append({
                    'class_name': class_name,
                    'confidence': conf_val,
                    'bbox': [x, y, w, h],
                    'center': [cx, cy],
                })

    except ImportError:
        pass  # ultralytics not available
    except Exception:
        pass  # Swallow detection errors; return empty list

    return detections


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _map_clock(h_zone: str, v_zone: str) -> int:
    """Return clock position from horizontal and vertical zone strings."""
    if h_zone == 'center':
        return 12
    if h_zone == 'right':
        return {'top': 1, 'middle': 2, 'bottom': 3}[v_zone]
    # left
    return {'top': 11, 'middle': 10, 'bottom': 9}[v_zone]


def _clock_position(cx: int, cy: int, frame_w: int, frame_h: int) -> int:
    """Compute clock position for a centre point."""
    nx = cx / frame_w
    ny = cy / frame_h
    h_zone = 'left' if nx < 0.33 else ('center' if nx < 0.67 else 'right')
    v_zone = 'top' if ny < 0.33 else ('middle' if ny < 0.67 else 'bottom')
    return _map_clock(h_zone, v_zone)


def _clock_label(pos: int) -> str:
    """Human-readable label for a clock position."""
    if pos == 12:
        return "12 o'clock"
    return f"{pos} o'clock"


# ---------------------------------------------------------------------------
# Distance helper (for picking the 'closest' outlet)
# ---------------------------------------------------------------------------

def _apparent_size(detection: Dict[str, Any]) -> float:
    """Return the area of the bounding box as a proxy for closeness."""
    _, _, w, h = detection['bbox']
    return w * h


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def _build_response(message: str, is_streaming: bool) -> Any:
    """Wrap a message in the standard audio-dict format."""
    if is_streaming:
        words = message.split()
        if len(words) > STREAMING_WORD_LIMIT:
            message = ' '.join(words[:STREAMING_WORD_LIMIT])
    return {
        'audio': {
            'type': 'speech',
            'text': message,
            'rate': 1.0,
        },
        'text': message,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(image: np.ndarray, input_data: Any = None) -> Any:
    """
    Detect electrical wall outlets and provide directional audio feedback.

    Args:
        image: Camera frame as a BGR numpy array (may be None).
        input_data: Optional config dict:
            - confidence (float): Detection threshold (default 0.20)
            - is_streaming (bool): Streaming mode flag (default False)

    Returns:
        Audio-friendly string or dict, or "" in streaming mode when nothing changed.

    Examples:
        One outlet slightly left  → "1 plug socket at 11 o'clock"
        Two outlets, nearest ahead → "2 plug sockets found, nearest at 12 o'clock"
        No outlets               → "no plug sockets found"
    """
    global _last_announced, _consecutive_no_socket_frames

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {'type': 'error', 'text': 'No camera image available', 'rate': 1.0},
            'text': 'No camera image available',
        }

    config = input_data if isinstance(input_data, dict) else {}
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    is_streaming = config.get('is_streaming', False)

    frame_h, frame_w = image.shape[:2]

    detections = detect_sockets(image, confidence)

    # --- No outlets found ---
    if not detections:
        _consecutive_no_socket_frames += 1

        if is_streaming:
            # Suppress repeated "nothing found" announcements
            if (_last_announced != 'none' and
                    _consecutive_no_socket_frames >= NO_SOCKET_FRAME_THRESHOLD):
                _last_announced = 'none'
                return _build_response('no plug sockets found', is_streaming)
            # Still waiting or already announced — stay silent.
            # Clear any previous socket-position state so that when sockets
            # reappear after a gap they are re-announced even if at the same
            # position as before.
            if _consecutive_no_socket_frames >= NO_SOCKET_FRAME_THRESHOLD:
                _last_announced = 'none'
            return ""

        # One-shot mode: reset streaming state so a subsequent streaming
        # session starts fresh, then always respond.
        _last_announced = None
        _consecutive_no_socket_frames = 0
        return _build_response('no plug sockets found', is_streaming)

    # --- Outlets found ---
    _consecutive_no_socket_frames = 0

    count = len(detections)

    # Pick the closest outlet (largest apparent size)
    closest = max(detections, key=_apparent_size)
    cx, cy = closest['center']
    clock_pos = _clock_position(cx, cy, frame_w, frame_h)
    pos_label = _clock_label(clock_pos)

    if count == 1:
        message = f"1 plug socket at {pos_label}"
    else:
        message = f"{count} plug sockets found, nearest at {pos_label}"

    if is_streaming:
        if message == _last_announced:
            return ""  # Nothing changed — stay silent
        _last_announced = message
    else:
        # One-shot mode: reset streaming state to avoid stale carry-over.
        _last_announced = None

    return _build_response(message, is_streaming)
