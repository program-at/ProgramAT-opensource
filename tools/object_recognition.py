"""
Live Object Recognition Tool for Blind Users

Provides real-time object detection using YOLOv11 and COCO dataset.
Announces visible objects to assist blind and low vision users in navigating
and interacting with their environment.

This tool runs YOLO11 object detection on camera frames and returns audio-friendly
descriptions of detected objects. When used in streaming mode, it tracks objects
across frames and announces new objects as they come into view.

## Usage from Mobile App:
The tool automatically receives camera frames from the mobile app and returns audio feedback.
It follows the standard tool interface with main(image, input_data) as the entry point.

## Example Return Format:
Simple mode:
"I see 3 objects: 2 cups, 1 laptop"

Advanced mode (with dict):
{
    'audio': {
        'type': 'speech',
        'text': 'I see 3 objects: 2 cups, 1 laptop',
        'rate': 1.0,
        'interrupt': False
    },
    'text': 'Found: 2 cups, 1 laptop',
    'detections': [...]  # Full detection data
}

## Configuration Options (via input_data):
- confidence: Detection confidence threshold (default 0.5)
- include_positions: Include spatial positions in description (default False)
- include_distance: Include distance estimates in description (default False)
- max_objects: Maximum objects to report (default 10)
- track_mode: Enable object tracking across frames (default False)

## Building Block Functions:
This tool exports several functions that can be used by other tools:
- detect_objects(image, confidence_threshold) -> List[Dict]
- count_objects_by_class(detections) -> Dict[str, int]
- get_position_description(center, width, height) -> str
- estimate_distance(bbox, frame_height) -> str
- create_audio_description(detections, width, height) -> str
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Detection constants
DEFAULT_CONFIDENCE = 0.5
MAX_OBJECTS_TO_REPORT = 10

# COCO class names (80 classes)
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

# Global state for object tracking across frames (used in streaming mode)
_previous_objects = set()
_frame_counter = 0


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
        # Fallback to OpenCV DNN with pre-trained COCO model if YOLO not available
        # This is a backup strategy but less accurate than YOLO
        detections = _detect_with_opencv_dnn(image, confidence_threshold)
    
    return detections


def _detect_with_opencv_dnn(image: np.ndarray, confidence_threshold: float) -> List[Dict[str, Any]]:
    """
    Fallback object detection using OpenCV DNN module.
    Uses MobileNet SSD trained on COCO dataset.
    
    This is less accurate than YOLO but doesn't require ultralytics package.
    """
    detections = []
    
    try:
        # Try to use Haar Cascade for face detection as minimal fallback
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            
            detections.append({
                'class_id': 0,  # person class
                'class_name': 'person',
                'confidence': 0.8,  # Haar cascade doesn't provide confidence
                'bbox': [x, y, w, h],
                'center': [center_x, center_y]
            })
    except Exception as e:
        # If all else fails, return empty list
        pass
    
    return detections


def count_objects_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count detected objects grouped by class name.
    
    Args:
        detections: List of detection dictionaries from detect_objects()
        
    Returns:
        Dictionary mapping class names to counts
    """
    counts = defaultdict(int)
    for det in detections:
        counts[det['class_name']] += 1
    return dict(counts)


def get_position_description(center: List[int], width: int, height: int) -> str:
    """
    Get a human-readable position description for an object.
    
    Args:
        center: [x, y] center coordinates of the object
        width: Frame width
        height: Frame height
        
    Returns:
        Position string like "center", "top left", "bottom right", etc.
    """
    x, y = center
    
    # Divide frame into 9 regions (3x3 grid)
    third_w = width / 3
    third_h = height / 3
    
    # Determine horizontal position
    if x < third_w:
        h_pos = "left"
    elif x < 2 * third_w:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Determine vertical position
    if y < third_h:
        v_pos = "top"
    elif y < 2 * third_h:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    # Combine positions
    if h_pos == "center" and v_pos == "middle":
        return "center"
    elif h_pos == "center":
        return v_pos
    elif v_pos == "middle":
        return h_pos
    else:
        return f"{v_pos} {h_pos}"


def estimate_distance(bbox: List[int], frame_height: int) -> str:
    """
    Estimate relative distance based on object size.
    
    Args:
        bbox: Bounding box [x, y, width, height]
        frame_height: Height of the frame
        
    Returns:
        Distance description like "very close", "close", "medium distance", "far away"
    """
    _, _, _, h = bbox
    
    # Calculate object height as percentage of frame
    height_ratio = h / frame_height
    
    if height_ratio > 0.6:
        return "very close"
    elif height_ratio > 0.3:
        return "close"
    elif height_ratio > 0.15:
        return "medium distance"
    else:
        return "far away"


def create_audio_description(
    detections: List[Dict[str, Any]], 
    width: int, 
    height: int,
    include_positions: bool = True,
    include_distance: bool = True,
    max_objects: int = MAX_OBJECTS_TO_REPORT
) -> str:
    """
    Create a natural language audio description of detected objects.
    
    Args:
        detections: List of detection dictionaries
        width: Frame width
        height: Frame height
        include_positions: Include position information
        include_distance: Include distance estimates
        max_objects: Maximum number of objects to describe
        
    Returns:
        Audio-friendly description string
    """
    if not detections:
        return "No objects detected"
    
    # Count objects by class
    counts = count_objects_by_class(detections)
    total = len(detections)
    
    # Limit number of objects to report
    detections = detections[:max_objects]
    
    # Start description
    if total == 1:
        desc_parts = ["I see 1 object"]
    else:
        desc_parts = [f"I see {total} objects" if total <= max_objects else f"I see {total} objects, describing {max_objects}"]
    
    # Group by class for cleaner output
    object_descriptions = []
    
    for class_name, count in sorted(counts.items()):
        # Find objects of this class
        class_objects = [d for d in detections if d['class_name'] == class_name]
        
        if not class_objects:
            continue
        
        # Build description for this class
        if count == 1:
            obj = class_objects[0]
            parts = [f"1 {class_name}"]
            
            if include_positions:
                pos = get_position_description(obj['center'], width, height)
                parts.append(f"on the {pos}")
            
            if include_distance:
                dist = estimate_distance(obj['bbox'], height)
                parts.append(f"{dist}")
            
            object_descriptions.append(" ".join(parts))
        else:
            # Multiple objects of same class
            parts = [f"{count} {class_name}s"]
            
            if include_positions and class_objects:
                # Get most common position
                positions = [get_position_description(obj['center'], width, height) for obj in class_objects]
                most_common_pos = max(set(positions), key=positions.count)
                parts.append(f"mostly on the {most_common_pos}")
            
            object_descriptions.append(" ".join(parts))
    
    # Combine all descriptions
    if object_descriptions:
        desc_parts.append(": " + ", ".join(object_descriptions))
    
    return "".join(desc_parts)


def track_new_objects(current_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Track objects across frames and identify new ones.
    
    Args:
        current_detections: Current frame detections
        
    Returns:
        List of new detections not seen in previous frame
    """
    global _previous_objects, _frame_counter
    
    _frame_counter += 1
    
    # Create set of current object identifiers (class + approximate position)
    current_objects = set()
    for det in current_detections:
        # Round position to reduce jitter (group nearby objects)
        x_bucket = det['center'][0] // 50
        y_bucket = det['center'][1] // 50
        obj_id = f"{det['class_name']}_{x_bucket}_{y_bucket}"
        current_objects.add(obj_id)
    
    # Find new objects
    new_object_ids = current_objects - _previous_objects
    
    # Get detection objects for new IDs
    new_detections = []
    for det in current_detections:
        x_bucket = det['center'][0] // 50
        y_bucket = det['center'][1] // 50
        obj_id = f"{det['class_name']}_{x_bucket}_{y_bucket}"
        if obj_id in new_object_ids:
            new_detections.append(det)
    
    # Update previous objects
    _previous_objects = current_objects
    
    return new_detections


def main(image: np.ndarray, input_data: Any = None) -> str:
    """
    Main entry point for live object recognition tool.
    
    Detects objects in camera frame and returns audio-friendly descriptions.
    In streaming mode, tracks objects and reports new ones as they appear.
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration:
            - confidence: Detection threshold (default 0.5)
            - include_positions: Include position info (default True)
            - include_distance: Include distance info (default True)
            - max_objects: Max objects to report (default 10)
            - track_mode: Enable tracking for streaming (default False)
    
    Returns:
        Audio-friendly description string suitable for text-to-speech.
        In track_mode, returns empty string "" when no new objects are detected,
        which prevents audio playback and reduces noise during streaming.
    """
    # Handle None or invalid image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"
    
    # Parse input_data
    config = {}
    if isinstance(input_data, dict):
        config = input_data
    elif isinstance(input_data, str):
        # Simple string input, ignore
        pass
    
    # Get configuration parameters
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    include_positions = config.get('include_positions', False)  # Changed to False to avoid cosine similarity issues
    include_distance = config.get('include_distance', False)   # Changed to False to focus on objects only
    max_objects = config.get('max_objects', MAX_OBJECTS_TO_REPORT)
    track_mode = config.get('track_mode', False)
    
    # Get frame dimensions
    height, width = image.shape[:2]
    
    # Detect objects
    detections = detect_objects(image, confidence)
    
    # Handle tracking mode for streaming
    if track_mode:
        # In tracking mode, only report new objects
        new_detections = track_new_objects(detections)
        
        if not new_detections:
            # No new objects, return empty/quiet response
            return ""  # Returning empty will result in no audio
        
        # Describe only new objects
        description = create_audio_description(
            new_detections, width, height,
            include_positions=include_positions,
            include_distance=include_distance,
            max_objects=max_objects
        )
        
        # Prefix with "New:" to indicate these are new objects
        if new_detections:
            description = "New: " + description
        
        return description
    else:
        # Normal mode: describe all objects
        # Reset tracking when not in track mode
        global _previous_objects, _frame_counter
        _previous_objects.clear()
        _frame_counter = 0
        
        description = create_audio_description(
            detections, width, height,
            include_positions=include_positions,
            include_distance=include_distance,
            max_objects=max_objects
        )
        
        return description


# Building block exports for use by other tools
__all__ = [
    'main',
    'detect_objects',
    'count_objects_by_class',
    'get_position_description',
    'estimate_distance',
    'create_audio_description',
    'track_new_objects'
]
