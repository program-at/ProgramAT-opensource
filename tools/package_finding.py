"""
Package Finding Assistance Tool for Blind Users

Helps blind users identify whether a package is visible in the camera view,
providing location information and directional guidance.

Uses YoloWorld for package detection since "package", "box", and "carton" 
are not in the COCO classes.

For still frames: Returns yes/no answer with package description and location
For streaming: Provides directional guidance when package detected

## Usage from Mobile App:
The tool automatically receives camera frames from the mobile app and returns audio feedback.

## Example Return Format (Still Frame):
"Yes, package found! Medium sized box near the left side of the frame"
"No package detected"

## Example Return Format (Streaming):
"No package found" (initially)
"Package detected! Move forward, medium box ahead"
"Getting closer, 2 feet away"

## Configuration Options (via input_data):
- mode: 'still' or 'stream' (default: 'still')
- confidence: Detection confidence threshold (default 0.4)
- package_classes: List of classes to detect (default: package, box, carton, parcel)

## Building Block Functions:
This tool uses positional and distance functions from object_recognition.py and camera_aiming.py

## Thread Safety Note:
This tool uses global state for streaming detection tracking (last detection and distance).
It is designed for single-user sequential frame processing (typical mobile app use case).

Model caching is handled by the backend server via a shared `yolo_model_cache` dictionary
that persists across all tool executions, enabling true real-time performance.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any

# Detection constants
DEFAULT_CONFIDENCE = 0.4  # Lower threshold for YoloWorld

# Package-related classes for YoloWorld detection
# Note: "package", "box", "carton" are NOT in COCO classes, so we use YoloWorld
DEFAULT_PACKAGE_CLASSES = ['package', 'box', 'carton', 'parcel']

# Global state for streaming mode (tool-specific state)
_last_detection = None
_last_distance_category = None


def detect_packages(
    image: np.ndarray, 
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    package_classes: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Detect packages in an image using YoloWorld.
    
    Since "package", "box", and "carton" are NOT in COCO classes,
    we use YoloWorld which can detect custom classes.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        package_classes: List of class names to detect (default: package, box, carton, parcel)
        
    Returns:
        List of detection dictionaries, each containing:
            - class_id: Integer class ID (0-based index from package_classes)
            - class_name: Human-readable class name
            - confidence: Detection confidence (0.0 to 1.0)
            - bbox: Bounding box [x, y, width, height]
            - center: Center point [x, y]
    """
    if image is None or image.size == 0:
        return []
    
    if package_classes is None:
        package_classes = DEFAULT_PACKAGE_CLASSES
    
    detections = []
    
    try:
        # Import ultralytics YOLO
        from ultralytics import YOLO
        
        # Get the shared model cache from the execution environment
        # This is injected by the backend server to persist across tool executions
        model_cache = globals().get('yolo_model_cache', {})
        
        # Cache key for this specific model configuration
        cache_key = 'package_finding_yolov8s_world'
        
        # Load YoloWorld model once and cache it in shared memory (MAJOR PERFORMANCE FIX)
        # This prevents reloading the model on every frame, making it truly real-time
        if cache_key not in model_cache:
            # First time: load model (will auto-download on first use)
            # Using YOLOv8-world for open-vocabulary detection
            model_cache[cache_key] = {
                'model': YOLO('yolov8s-world.pt'),
                'classes': None
            }
        
        model_info = model_cache[cache_key]
        yolo_model = model_info['model']
        
        # Set custom classes if they changed (or first time)
        classes_key = tuple(package_classes)  # Convert to tuple for hashable comparison
        if model_info['classes'] is None or model_info['classes'] != classes_key:
            yolo_model.set_classes(package_classes)
            model_info['classes'] = classes_key
        
        # Run inference (now fast since model is already loaded)
        results = yolo_model(image, conf=confidence_threshold, verbose=False)
        
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
                
                # Get class name from our custom classes
                class_name = package_classes[class_id] if class_id < len(package_classes) else "package"
                
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
        # YoloWorld not available
        pass
    except Exception as e:
        # Log error but don't crash
        pass
    
    return detections


def get_position_description(center: List[int], width: int, height: int) -> str:
    """
    Get a human-readable position description for an object.
    Copied from object_recognition.py as building block.
    
    Args:
        center: [x, y] center coordinates of the object
        width: Frame width
        height: Frame height
        
    Returns:
        Position string like "center", "left", "right", "top left", etc.
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
    
    # Combine positions (match object_recognition.py exactly)
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
    Copied from object_recognition.py as building block.
    
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


def get_size_description(bbox: List[int], frame_height: int, frame_width: int) -> str:
    """
    Get a description of the package size.
    
    Args:
        bbox: Bounding box [x, y, width, height]
        frame_height: Height of the frame
        frame_width: Width of the frame
        
    Returns:
        Size description like "small", "medium", "large"
    """
    _, _, w, h = bbox
    
    # Calculate area as percentage of frame
    area_ratio = (w * h) / (frame_width * frame_height)
    
    if area_ratio > 0.4:
        return "large"
    elif area_ratio > 0.15:
        return "medium"
    else:
        return "small"


def get_directional_guidance(center: List[int], width: int, height: int, distance: str) -> str:
    """
    Generate directional guidance for moving toward the package.
    
    Args:
        center: [x, y] center coordinates of the package
        width: Frame width
        height: Frame height
        distance: Distance category from estimate_distance()
        
    Returns:
        Directional guidance string
    """
    x, y = center
    
    # Horizontal guidance
    h_offset = (x / width) - 0.5  # -0.5 to 0.5
    
    if abs(h_offset) < 0.15:
        h_direction = "straight ahead"
    elif h_offset > 0.3:
        h_direction = "to the right"
    elif h_offset > 0:
        h_direction = "slightly to the right"
    elif h_offset < -0.3:
        h_direction = "to the left"
    else:
        h_direction = "slightly to the left"
    
    # Combine with distance
    if distance == "very close":
        return f"Package is {h_direction}, very close, within reach"
    elif distance == "close":
        return f"Package is {h_direction}, about 2 to 3 feet away"
    elif distance == "medium distance":
        return f"Move forward, package {h_direction}, about 5 to 10 feet away"
    else:
        return f"Move forward, package {h_direction}, more than 10 feet away"


def find_packages(
    detections: List[Dict[str, Any]], 
    package_classes: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter detections to only include package-like objects.
    
    Note: Since we're using YoloWorld with custom classes,
    all detections should already be packages. This function
    is kept for consistency with the building block pattern.
    
    Args:
        detections: List of all detected objects
        package_classes: List of class names to treat as packages (optional filter)
        
    Returns:
        List of package detections
    """
    if package_classes is None:
        # All detections are already packages from YoloWorld
        return detections
    
    # Filter to specific package classes if requested
    packages = [d for d in detections if d['class_name'] in package_classes]
    return packages


def create_still_frame_response(
    packages: List[Dict[str, Any]], 
    width: int, 
    height: int
) -> str:
    """
    Create response for still frame mode.
    
    Args:
        packages: List of detected packages
        width: Frame width
        height: Frame height
        
    Returns:
        Audio-friendly yes/no response with details
    """
    if not packages:
        return "No package detected"
    
    # Get the most prominent package (largest)
    package = max(packages, key=lambda p: p['bbox'][2] * p['bbox'][3])
    
    # Build description
    size = get_size_description(package['bbox'], height, width)
    position = get_position_description(package['center'], width, height)
    package_type = package['class_name']
    
    # Create natural response
    if len(packages) == 1:
        response = f"Yes, package found! {size.capitalize()} {package_type} near the {position}"
    else:
        response = f"Yes, {len(packages)} packages found! Largest is a {size} {package_type} near the {position}"
    
    return response


def create_streaming_response(
    packages: List[Dict[str, Any]], 
    width: int, 
    height: int
) -> str:
    """
    Create response for streaming mode with directional guidance.
    
    Args:
        packages: List of detected packages
        width: Frame width
        height: Frame height
        
    Returns:
        Audio-friendly directional guidance or "no package" message
    """
    global _last_detection, _last_distance_category
    
    if not packages:
        # Reset state when no package
        _last_detection = None
        _last_distance_category = None
        return "No package found"
    
    # Get the most prominent package (largest)
    package = max(packages, key=lambda p: p['bbox'][2] * p['bbox'][3])
    
    # Get distance and direction
    distance = estimate_distance(package['bbox'], height)
    direction = get_directional_guidance(package['center'], width, height, distance)
    size = get_size_description(package['bbox'], height, width)
    package_type = package['class_name']
    
    # Check if this is a new detection or distance changed
    current_detection = f"{package['class_name']}_{size}"
    
    if _last_detection != current_detection:
        # New package detected
        _last_detection = current_detection
        _last_distance_category = distance
        return f"Package detected! {size.capitalize()} {package_type}. {direction}"
    elif _last_distance_category != distance:
        # Same package, but distance changed
        _last_distance_category = distance
        return direction
    else:
        # Same package, same distance - don't repeat
        return ""


def main(image: np.ndarray, input_data: Any = None) -> str:
    """
    Main entry point for package finding assistance tool.
    
    Detects packages in camera frame using YoloWorld and provides 
    appropriate feedback based on mode (still frame vs streaming).
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration:
            - mode: 'still' or 'stream' (default 'still')
            - confidence: Detection threshold (default 0.4)
            - package_classes: List of classes to detect (default: package, box, carton, parcel)
    
    Returns:
        Audio-friendly response suitable for text-to-speech.
        In streaming mode, returns empty string "" when no status change,
        which prevents redundant audio playback.
    """
    # Handle None or invalid image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"
    
    # Parse input_data
    config = {}
    if isinstance(input_data, dict):
        config = input_data
    elif isinstance(input_data, str):
        # Try to parse mode from string
        if input_data.lower() in ['still', 'stream']:
            config = {'mode': input_data.lower()}
    
    # Get configuration parameters
    mode = config.get('mode', 'still')
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    package_classes = config.get('package_classes', DEFAULT_PACKAGE_CLASSES)
    
    # Get frame dimensions
    height, width = image.shape[:2]
    
    # Detect packages using YoloWorld
    packages = detect_packages(image, confidence, package_classes)
    
    # Generate response based on mode
    if mode == 'stream':
        return create_streaming_response(packages, width, height)
    else:
        return create_still_frame_response(packages, width, height)


# Building block exports for use by other tools
__all__ = [
    'main',
    'detect_packages',
    'get_position_description',
    'estimate_distance',
    'get_size_description',
    'get_directional_guidance',
    'find_packages',
    'create_still_frame_response',
    'create_streaming_response'
]
