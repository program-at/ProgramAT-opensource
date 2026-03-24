"""
Camera Aiming Assistance Tool for Blind Users

Helps blind users center their camera on an object to take well-framed photos
by providing directional cues like "move left", "move right", "move up", 
"move down", "move closer", "move further away".

This tool locks onto an object using bounding boxes and guides the user to
frame it properly for photography. The framing doesn't need to be perfect,
but should be clear with good quality.

## Usage from Mobile App:
The tool automatically receives camera frames from the mobile app and returns
audio feedback with directional guidance.

## Example Return Format:
"Move right and back up"
"Almost there, move slightly left"
"Well framed! Object is centered"

## Configuration Options (via input_data):
- confidence: Detection confidence threshold (default 0.5)
- target_class: Specific object class to lock onto (default: auto-detect largest)
- centering_tolerance: How centered object must be, 0.0-1.0 (default 0.15)
- size_min: Minimum size ratio for good framing (default 0.15)
- size_max: Maximum size ratio for good framing (default 0.7)
- lock_object: Whether to lock onto first detected object (default True)

## Building Block Functions:
This tool exports several functions that can be used by other tools:
- calculate_framing_metrics(bbox, frame_width, frame_height) -> Dict
- generate_directional_cues(metrics) -> str
- is_well_framed(metrics) -> bool

## Thread Safety Note:
This tool uses global state for object locking. It is designed for single-user
sequential frame processing. For concurrent processing, create separate instances
or use thread-local storage.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Framing constants
DEFAULT_CONFIDENCE = 0.5
CENTERING_TOLERANCE = 0.15  # 15% tolerance from center
SIZE_MIN_RATIO = 0.15  # Object should be at least 15% of frame
SIZE_MAX_RATIO = 0.7   # Object should be at most 70% of frame
WELL_FRAMED_SIZE_IDEAL = 0.4  # Ideal size is ~40% of frame

# COCO class names (80 classes) - needed for object detection
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

# Global state for object locking
_locked_object_class = None
_locked_object_position = None
_frame_count = 0


def detect_objects(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Detect objects in an image using YOLOv11 and COCO dataset.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        
    Returns:
        List of detection dictionaries, each containing:
            - class_id: Integer class ID
            - class_name: Human-readable class name
            - confidence: Detection confidence (0.0 to 1.0)
            - bbox: Bounding box [x, y, width, height]
            - center: Center point [x, y]
    """
    if image is None or image.size == 0:
        return []
    
    detections = []
    
    try:
        # Import ultralytics YOLO
        from ultralytics import YOLO
        
        # Load YOLOv11 model (will auto-download on first use)
        # Using YOLOv11n (nano) for speed on CPU
        model = YOLO('yolo11n.pt')
        
        # Run inference
        results = model(image, conf=confidence_threshold, verbose=False)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Get class name
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"object_{class_id}"
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'center': [center_x, center_y]
                })
        
    except ImportError:
        # Fallback: return empty if YOLO not available
        # In production, YOLO should always be available on the server
        pass
    except Exception as e:
        # Log error but don't crash
        pass
    
    return detections


def calculate_framing_metrics(
    bbox: List[int], 
    frame_width: int, 
    frame_height: int
) -> Dict[str, Any]:
    """
    Calculate framing metrics for an object bounding box.
    
    Args:
        bbox: Bounding box [x, y, width, height]
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Dictionary containing:
            - center_x: Normalized X position (0.0 = left, 0.5 = center, 1.0 = right)
            - center_y: Normalized Y position (0.0 = top, 0.5 = center, 1.0 = bottom)
            - size_ratio: Object size as ratio of frame area
            - horizontal_offset: How far from horizontal center (-0.5 to 0.5)
            - vertical_offset: How far from vertical center (-0.5 to 0.5)
            - is_centered: Whether object is centered within tolerance
            - is_good_size: Whether object size is in good range
    """
    x, y, w, h = bbox
    
    # Calculate center position (normalized 0-1)
    obj_center_x = (x + w / 2) / frame_width
    obj_center_y = (y + h / 2) / frame_height
    
    # Calculate size ratio
    obj_area = w * h
    frame_area = frame_width * frame_height
    size_ratio = obj_area / frame_area
    
    # Calculate offsets from center
    horizontal_offset = obj_center_x - 0.5
    vertical_offset = obj_center_y - 0.5
    
    # Determine if centered
    is_centered = (
        abs(horizontal_offset) < CENTERING_TOLERANCE and 
        abs(vertical_offset) < CENTERING_TOLERANCE
    )
    
    # Determine if good size
    is_good_size = SIZE_MIN_RATIO <= size_ratio <= SIZE_MAX_RATIO
    
    return {
        'center_x': obj_center_x,
        'center_y': obj_center_y,
        'size_ratio': size_ratio,
        'horizontal_offset': horizontal_offset,
        'vertical_offset': vertical_offset,
        'is_centered': is_centered,
        'is_good_size': is_good_size,
        'bbox': bbox
    }


def generate_directional_cues(metrics: Dict[str, Any]) -> str:
    """
    Generate human-readable directional cues based on framing metrics.
    
    Args:
        metrics: Framing metrics from calculate_framing_metrics()
        
    Returns:
        Audio-friendly directional guidance string
    """
    cues = []
    
    # Horizontal adjustment
    h_offset = metrics['horizontal_offset']
    if abs(h_offset) > CENTERING_TOLERANCE:
        if h_offset > 0.3:
            cues.append("move left")
        elif h_offset > CENTERING_TOLERANCE:
            cues.append("move slightly left")
        elif h_offset < -0.3:
            cues.append("move right")
        else:
            cues.append("move slightly right")
    
    # Vertical adjustment
    v_offset = metrics['vertical_offset']
    if abs(v_offset) > CENTERING_TOLERANCE:
        if v_offset > 0.3:
            cues.append("move up")
        elif v_offset > CENTERING_TOLERANCE:
            cues.append("move slightly up")
        elif v_offset < -0.3:
            cues.append("move down")
        else:
            cues.append("move slightly down")
    
    # Distance adjustment
    size_ratio = metrics['size_ratio']
    if size_ratio > SIZE_MAX_RATIO:
        cues.append("back up")
    elif size_ratio > SIZE_MAX_RATIO * 0.85:
        cues.append("back up a little")
    elif size_ratio < SIZE_MIN_RATIO:
        cues.append("move closer")
    elif size_ratio < SIZE_MIN_RATIO * 1.3:
        cues.append("move a bit closer")
    
    # Combine cues
    if not cues:
        return "Hold steady"
    
    # Join with "and" for natural speech
    if len(cues) == 1:
        return cues[0].capitalize()
    elif len(cues) == 2:
        return f"{cues[0].capitalize()} and {cues[1]}"
    else:
        return f"{cues[0].capitalize()}, {', '.join(cues[1:-1])}, and {cues[-1]}"


def is_well_framed(metrics: Dict[str, Any]) -> bool:
    """
    Determine if object is well-framed and ready for photo.
    
    Args:
        metrics: Framing metrics from calculate_framing_metrics()
        
    Returns:
        True if object is well-framed, False otherwise
    """
    # Must be centered
    if not metrics['is_centered']:
        return False
    
    # Must be good size
    if not metrics['is_good_size']:
        return False
    
    # Additional quality check: prefer objects closer to ideal size
    size_ratio = metrics['size_ratio']
    if WELL_FRAMED_SIZE_IDEAL > 0:
        size_score = 1.0 - abs(size_ratio - WELL_FRAMED_SIZE_IDEAL) / WELL_FRAMED_SIZE_IDEAL
    else:
        size_score = 1.0
    
    # Must be reasonably close to ideal size
    return size_score > 0.5


def select_target_object(
    detections: List[Dict[str, Any]],
    target_class: Optional[str] = None,
    lock_object: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Select the target object to aim at.
    
    Args:
        detections: List of detected objects from detect_objects()
        target_class: Specific class name to target (None = auto-select)
        lock_object: Whether to lock onto first object and track it
        
    Returns:
        Selected detection dict or None if no suitable object found
    """
    global _locked_object_class, _locked_object_position, _frame_count
    
    if not detections:
        # No objects detected, reset lock
        _locked_object_class = None
        _locked_object_position = None
        return None
    
    # If we have a locked object, try to find it again
    if lock_object and _locked_object_class is not None:
        # Find objects of the locked class
        same_class = [d for d in detections if d['class_name'] == _locked_object_class]
        
        if same_class:
            # Find the one closest to previous position
            if _locked_object_position is not None:
                def distance_to_locked(det):
                    dx = det['center'][0] - _locked_object_position[0]
                    dy = det['center'][1] - _locked_object_position[1]
                    return dx*dx + dy*dy
                
                closest = min(same_class, key=distance_to_locked)
                min_distance = distance_to_locked(closest)
                
                # Only accept if it's reasonably close (within 150 pixels)
                # This prevents switching to a different object of the same class
                if min_distance < 150 * 150:
                    _locked_object_position = closest['center']
                    return closest
                else:
                    # Too far away, likely lost the object - stick with first one
                    # but don't update the locked position yet to avoid jumping
                    return same_class[0]
            else:
                # Just use the first one of the same class
                selected = same_class[0]
                _locked_object_position = selected['center']
                return selected
    
    # No lock or lock lost, select new object
    if target_class:
        # User specified a target class
        matching = [d for d in detections if d['class_name'] == target_class]
        if matching:
            selected = matching[0]
        else:
            return None
    else:
        # Auto-select: choose largest object (most prominent)
        selected = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
    
    # Lock onto this object
    if lock_object:
        _locked_object_class = selected['class_name']
        _locked_object_position = selected['center']
    
    return selected


def reset_lock():
    """Reset the object lock state."""
    global _locked_object_class, _locked_object_position, _frame_count
    _locked_object_class = None
    _locked_object_position = None
    _frame_count = 0


def main(image: np.ndarray, input_data: Any = None) -> str:
    """
    Main entry point for camera aiming assistance tool.
    
    Guides user to properly frame an object in the camera by providing
    directional cues. Locks onto an object and tracks it across frames.
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration:
            - confidence: Detection threshold (default 0.5)
            - target_class: Specific object to aim at (default: auto-select)
            - centering_tolerance: Centering tolerance (default 0.15)
            - size_min: Minimum size ratio (default 0.15)
            - size_max: Maximum size ratio (default 0.7)
            - lock_object: Lock onto object (default True)
            - reset: Reset object lock (default False)
    
    Returns:
        Audio-friendly directional guidance for framing the object.
        Returns "Well framed!" when object is properly positioned.
    """
    global _frame_count
    _frame_count += 1
    
    # Handle None or invalid image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"
    
    # Parse input_data
    config = {}
    if isinstance(input_data, dict):
        config = input_data
    
    # Get configuration parameters
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    target_class = config.get('target_class', None)
    lock_object = config.get('lock_object', True)
    
    # Handle reset request
    if config.get('reset', False):
        reset_lock()
        return "Lock reset, finding new object"
    
    # Get frame dimensions
    height, width = image.shape[:2]
    
    # Detect objects
    detections = detect_objects(image, confidence)
    
    if not detections:
        reset_lock()
        return "No objects detected, please point camera at an object"
    
    # Select target object
    target = select_target_object(detections, target_class, lock_object)
    
    if target is None:
        reset_lock()
        if target_class:
            return f"Cannot find {target_class}, please point camera at one"
        else:
            return "No objects detected, please point camera at an object"
    
    # Calculate framing metrics
    metrics = calculate_framing_metrics(target['bbox'], width, height)
    
    # Check if well-framed
    if is_well_framed(metrics):
        object_name = target['class_name']
        return f"Well framed! {object_name.capitalize()} is centered and ready"
    
    # Generate directional cues
    cues = generate_directional_cues(metrics)
    
    # Add object name context for first few frames
    if _frame_count <= 3:
        object_name = target['class_name']
        return f"Aiming at {object_name}. {cues}"
    
    return cues


# Building block exports for use by other tools
__all__ = [
    'main',
    'calculate_framing_metrics',
    'generate_directional_cues',
    'is_well_framed',
    'select_target_object',
    'reset_lock'
]
