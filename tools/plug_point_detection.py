"""
Plug Point and Electrical Outlet Detection

A streaming tool that identifies visible plug points and electrical outlets
using YoloWorld, counts them, and provides clock face directions.

Features:
- Detects plug points/outlets using YoloWorld (not in COCO classes)
- Counts all visible outlets in the frame
- Provides clock face positions (1-3 and 9-12 only)
- Streaming mode limited to 15 words
- Only announces when count or position changes (reduces noise)

Navigation System:
- Uses clock face positions (1-3 o'clock on right, 9-12 o'clock on left)
- Positions 4-8 not used (would be behind the camera)

This tool runs on the backend server and receives camera frames from the mobile app.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional

# Detection constants
DEFAULT_CONFIDENCE = 0.20  # Low threshold: outlets can be small and varied in appearance

OUTLET_CLASSES = [
    'electrical outlet', 'power outlet', 'wall outlet', 'plug socket',
    'power socket', 'electrical socket', 'plug point', 'wall socket',
    'power strip', 'extension socket', 'mains socket', 'receptacle',
    'electric plug', 'outlet plate', 'socket plate',
]

# Only valid camera-visible clock positions
VALID_CLOCK_POSITIONS = [12, 1, 2, 3, 9, 10, 11]

# Streaming word limit
STREAMING_WORD_LIMIT = 15

# Temporal smoothing: require N consecutive frames with no outlets before announcing absence
NO_OUTLET_ANNOUNCEMENT_DELAY = 5

# Global streaming state
_last_outlet_count = -1          # -1 means "not yet announced"
_last_outlet_positions = None    # frozenset of clock positions reported last time
_consecutive_no_outlet_frames = 0


def detect_outlets(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Detect electrical outlets/plug points using YoloWorld.

    Args:
        image: Input image as numpy array (BGR format).
        confidence_threshold: Minimum detection confidence (0.0–1.0).

    Returns:
        List of detection dicts with keys: class_name, confidence, bbox, center.
    """
    if image is None or image.size == 0:
        return []

    detections = []

    try:
        from ultralytics import YOLO

        model_cache = globals().get('yolo_model_cache', {})
        cache_key = 'plug_point_detection_yolov8s_world'

        if cache_key not in model_cache:
            model_cache[cache_key] = {
                'model': YOLO('yolov8s-world.pt'),
                'classes': None
            }
            globals()['yolo_model_cache'] = model_cache

        model_info = model_cache[cache_key]
        yolo_model = model_info['model']

        classes_key = tuple(OUTLET_CLASSES)
        if model_info['classes'] != classes_key:
            yolo_model.set_classes(OUTLET_CLASSES)
            model_info['classes'] = classes_key

        results = yolo_model(image, conf=confidence_threshold, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = OUTLET_CLASSES[class_id] if 0 <= class_id < len(OUTLET_CLASSES) else "outlet"
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
    Map (x, y) to a camera-visible clock face position (1-3, 9-12).

    Args:
        center_x: X coordinate of object center.
        center_y: Y coordinate of object center.
        frame_width: Image width in pixels.
        frame_height: Image height in pixels.

    Returns:
        Clock position integer from {9, 10, 11, 12, 1, 2, 3}.
    """
    norm_x = center_x / frame_width
    norm_y = center_y / frame_height

    if norm_x < 0.33:
        h_region = 'left'
    elif norm_x < 0.67:
        h_region = 'center'
    else:
        h_region = 'right'

    if norm_y < 0.33:
        v_region = 'top'
    elif norm_y < 0.67:
        v_region = 'middle'
    else:
        v_region = 'bottom'

    if h_region == 'center' and v_region == 'top':
        return 12
    if h_region == 'right':
        if v_region == 'top':
            return 1
        elif v_region == 'middle':
            return 2
        else:  # bottom
            return 3
    if h_region == 'left':
        if v_region == 'top':
            return 11
        elif v_region == 'middle':
            return 10
        else:  # bottom
            return 9
    # Center-middle or center-bottom → treat as 12
    return 12


def build_response(count: int, detections: List[Dict[str, Any]],
                   frame_width: int, frame_height: int,
                   is_streaming: bool) -> str:
    """
    Build an audio-friendly response string describing detected outlets.

    Args:
        count: Number of detected outlets.
        detections: List of detection dicts (used for positions).
        frame_width: Image width.
        frame_height: Image height.
        is_streaming: Whether tool is in streaming mode.

    Returns:
        Audio-friendly string (≤15 words in streaming mode).
    """
    if count == 0:
        return "No plug points detected"

    # Gather unique clock positions
    clock_positions = []
    for det in detections:
        cx, cy = det['center']
        pos = get_clock_position(cx, cy, frame_width, frame_height)
        if pos not in clock_positions:
            clock_positions.append(pos)
    clock_positions.sort()

    # Build position string
    if len(clock_positions) == 1:
        pos_str = f"at {clock_positions[0]} o'clock"
    else:
        pos_parts = [f"{p} o'clock" for p in clock_positions]
        pos_str = "at " + " and ".join(pos_parts)

    outlet_word = "plug point" if count == 1 else "plug points"
    response = f"{count} {outlet_word} {pos_str}"

    if is_streaming:
        words = response.split()
        if len(words) > STREAMING_WORD_LIMIT:
            response = " ".join(words[:STREAMING_WORD_LIMIT])

    return response


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """
    Main entry point for plug point / electrical outlet detection.

    Detects outlets visible in the camera frame and returns a count with
    clock face directions suitable for text-to-speech output.

    Args:
        image: Camera frame as numpy array (BGR format from OpenCV).
        input_data: Optional configuration dict:
            - confidence (float): Detection threshold (default 0.20).
            - is_streaming (bool): Enable streaming mode (default False).

    Returns:
        Audio-friendly string, or empty string "" in streaming mode when
        nothing has changed (suppresses repeated TTS announcements).
    """
    global _last_outlet_count, _last_outlet_positions, _consecutive_no_outlet_frames

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"

    config = input_data if isinstance(input_data, dict) else {}
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    is_streaming = config.get('is_streaming', False)

    height, width = image.shape[:2]

    detections = detect_outlets(image, confidence)
    count = len(detections)

    if count == 0:
        _consecutive_no_outlet_frames += 1

        if is_streaming:
            # Suppress "no outlets" until we've had several consecutive empty frames
            if (_last_outlet_count == 0 or
                    _consecutive_no_outlet_frames < NO_OUTLET_ANNOUNCEMENT_DELAY):
                return ""
            _last_outlet_count = 0
            _last_outlet_positions = frozenset()
            return "No plug points detected"

        return "No plug points detected"

    # Outlets detected — reset no-outlet counter
    _consecutive_no_outlet_frames = 0

    # Compute clock positions for change detection
    current_positions = frozenset(
        get_clock_position(d['center'][0], d['center'][1], width, height)
        for d in detections
    )

    if is_streaming:
        # Skip announcement if count and positions haven't changed
        if count == _last_outlet_count and current_positions == _last_outlet_positions:
            return ""

    _last_outlet_count = count
    _last_outlet_positions = current_positions

    return build_response(count, detections, width, height, is_streaming)
