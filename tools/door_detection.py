"""
Door and Frame Detection with Navigation Aid

A tool that provides auditory (beep) and haptic (vibration) feedback for door and 
frame presence, along with verbal navigation instructions using clock face navigation.

Features:
- Detects doors and door frames using YoloWorld (since door is not in COCO classes)
- Provides beep sound and haptic vibration for door/frame presence
- Gives verbal navigation instructions using clock face positions (1-3 and 9-12)
- Audio-optimized output for text-to-speech
- Streaming mode limited to 15 words

Navigation System:
- Uses clock face positions (1-3 o'clock on right, 9-12 o'clock on left)
- Positions 4-8 not used as they would be behind the camera
- Haptic feedback via 'success' type in audio suite
- Beep patterns for quick proximity alerts

Example Usage:
User approaches a doorway and activates the tool. The tool detects and reports:
"Door detected at 12 o'clock, move forward" with haptic feedback

Model caching is handled by the backend server via a shared `yolo_model_cache` dictionary
that persists across all tool executions, enabling true real-time performance.

This tool runs on the backend server and receives camera frames from the mobile app.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Detection constants
# Balanced threshold for door detection - YoloWorld may produce lower confidence
# scores for non-COCO objects, and doors can have highly variable appearances
DEFAULT_CONFIDENCE = 0.25  # Balanced: 0.2 too sensitive, 0.3 too restrictive
DOOR_CLASSES = [
    # Complete doors
    'door', 'door frame', 'doorway', 'entrance',
    'open door', 'closed door', 'wooden door', 'glass door',
    'front door', 'entry', 'exit', 'portal',
    # Door parts for close-up detection when zoomed in
    'door panel', 'door handle', 'door knob', 'doorframe',
    'door edge', 'door surface', 'wall door', 'room entrance',
    'vertical door', 'rectangular door'
]

# Navigation constants
# Only use clock positions visible from camera (1-3, 9-12)
VALID_CLOCK_POSITIONS = [12, 1, 2, 3, 9, 10, 11]

# Distance thresholds based on bounding box size
DISTANCE_VERY_CLOSE = 0.6  # Object takes up 60%+ of frame height
DISTANCE_CLOSE = 0.3       # Object takes up 30%+ of frame height
DISTANCE_MEDIUM = 0.15     # Object takes up 15%+ of frame height

# Proximity inference constants
INFERRED_CONFIDENCE = 0.9  # High confidence for inferred detections
PROXIMITY_FRAME_WINDOW = 7  # Infer proximity if door seen within this many frames (increased from 5)
RESET_FRAME_THRESHOLD = 10  # Reset tracking after this many frames without detection

# Temporal smoothing to prevent transient "no door" messages
NO_DOOR_ANNOUNCEMENT_DELAY = 5  # Require this many consecutive frames without door before announcing

# Streaming mode word limit
STREAMING_WORD_LIMIT = 15

# Global state for streaming mode (tool-specific state)
_last_door_detection = None  # None, 'no_door', or position description
_last_door_count = 0
_last_door_bbox = None  # Track last detected door bounding box for proximity inference
_frames_since_door = 0  # Count frames since last door detection
_consecutive_no_door_frames = 0  # Count consecutive frames without detection for temporal smoothing


def detect_doors(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Detect doors and door frames using YoloWorld.
    
    Since 'door' and 'door frame' are not in the standard COCO dataset,
    we use YoloWorld which can detect arbitrary objects by text prompt.
    
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
        # Import ultralytics YOLO for YoloWorld
        from ultralytics import YOLO
        
        # Get the shared model cache from the execution environment
        # This is injected by the backend server to persist across tool executions
        model_cache = globals().get('yolo_model_cache', {})
        
        # Cache key for this specific model configuration
        cache_key = 'door_detection_yolov8s_world'
        
        # Load YoloWorld model once and cache it in shared memory
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
        classes_key = tuple(DOOR_CLASSES)  # Convert to tuple for hashable comparison
        if model_info['classes'] is None or model_info['classes'] != classes_key:
            yolo_model.set_classes(DOOR_CLASSES)
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
                
                # Get class name (with bounds checking)
                class_name = DOOR_CLASSES[class_id] if 0 <= class_id < len(DOOR_CLASSES) else "door"
                
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
        # YoloWorld not available - return empty
        # This can happen if ultralytics is not installed
        print("Warning: ultralytics not available for door detection")
    except Exception as e:
        # Log error but don't crash
        # This helps with debugging while keeping the tool robust
        print(f"Warning: Door detection error: {e}")
    
    return detections


def get_clock_position(center_x: int, center_y: int, frame_width: int, frame_height: int) -> int:
    """
    Determine clock face position of an object.
    
    Only returns positions visible from camera perspective:
    - 12: Top center
    - 1-3: Right side (top to bottom)
    - 9-11: Left side (top to bottom)
    
    Args:
        center_x: X coordinate of object center
        center_y: Y coordinate of object center
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Clock position (1-3, 9-12)
    """
    # Normalize coordinates (0-1)
    norm_x = center_x / frame_width
    norm_y = center_y / frame_height
    
    # Divide into regions
    # Horizontal: left (0-0.33), center (0.33-0.67), right (0.67-1.0)
    # Vertical: top (0-0.33), middle (0.33-0.67), bottom (0.67-1.0)
    
    # Determine horizontal region
    if norm_x < 0.33:
        h_region = 'left'
    elif norm_x < 0.67:
        h_region = 'center'
    else:
        h_region = 'right'
    
    # Determine vertical region
    if norm_y < 0.33:
        v_region = 'top'
    elif norm_y < 0.67:
        v_region = 'middle'
    else:
        v_region = 'bottom'
    
    # Map to clock positions
    # 12 o'clock: top center
    if h_region == 'center' and v_region == 'top':
        return 12
    
    # Right side (1-3)
    if h_region == 'right':
        if v_region == 'top':
            return 1
        elif v_region == 'middle':
            return 2
        else:  # bottom
            return 3
    
    # Left side (9-11)
    if h_region == 'left':
        if v_region == 'top':
            return 11
        elif v_region == 'middle':
            return 10
        else:  # bottom
            return 9
    
    # Center positions (not top)
    if h_region == 'center':
        if v_region == 'middle':
            return 12  # Straight ahead
        else:  # bottom
            return 12  # Treat as center
    
    # Default to 12
    return 12


def estimate_door_distance(bbox: List[int], frame_height: int) -> str:
    """
    Estimate relative distance to door based on bounding box size.
    
    Args:
        bbox: Bounding box [x, y, width, height]
        frame_height: Height of the frame
        
    Returns:
        Distance description: "very close", "close", "medium", or "far"
    """
    _, _, _, h = bbox
    
    # Calculate object height as percentage of frame
    height_ratio = h / frame_height
    
    if height_ratio > DISTANCE_VERY_CLOSE:
        return "very close"
    elif height_ratio > DISTANCE_CLOSE:
        return "close"
    elif height_ratio > DISTANCE_MEDIUM:
        return "medium"
    else:
        return "far"


def generate_navigation_instruction(
    detection: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    is_streaming: bool = False
) -> str:
    """
    Generate navigation instruction for reaching the detected door.
    
    Args:
        detection: Detection dictionary with bbox and center
        frame_width: Frame width
        frame_height: Frame height
        is_streaming: Whether in streaming mode (limits to 15 words)
        
    Returns:
        Navigation instruction string
    """
    center_x, center_y = detection['center']
    clock_pos = get_clock_position(center_x, center_y, frame_width, frame_height)
    distance = estimate_door_distance(detection['bbox'], frame_height)
    
    # Build instruction
    parts = []
    
    # Add clock position
    if clock_pos == 12:
        parts.append("straight ahead")
    elif clock_pos in [1, 2, 3]:
        parts.append(f"at {clock_pos} o'clock")
    elif clock_pos in [9, 10, 11]:
        parts.append(f"at {clock_pos} o'clock")
    
    # Add distance-based action
    if distance == "very close":
        parts.append("reach forward")
    elif distance == "close":
        parts.append("move forward")
    elif distance == "medium":
        parts.append("walk forward")
    else:  # far
        parts.append("walk ahead")
    
    instruction = ", ".join(parts)
    
    # Enforce word limit for streaming mode
    if is_streaming:
        words = instruction.split()
        if len(words) > STREAMING_WORD_LIMIT:
            instruction = " ".join(words[:STREAMING_WORD_LIMIT])
    
    return instruction


def create_door_detection_response(
    detections: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    is_streaming: bool = False,
    use_haptic: bool = True
) -> Dict[str, Any]:
    """
    Create audio response for door detection with haptic/beep feedback.
    
    Args:
        detections: List of door detections
        frame_width: Frame width
        frame_height: Frame height
        is_streaming: Whether in streaming mode (limits to 15 words)
        use_haptic: Whether to use haptic feedback
        
    Returns:
        Dictionary with audio configuration including haptic/beep feedback
    """
    if not detections:
        return {
            'audio': {
                'type': 'speech',
                'text': 'No door detected',
                'rate': 1.0
            },
            'text': 'No door detected'
        }
    
    # Use the most prominent detection (largest)
    main_detection = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
    
    # Generate navigation instruction
    nav_instruction = generate_navigation_instruction(
        main_detection, frame_width, frame_height, is_streaming
    )
    
    # Build text message
    count = len(detections)
    if count == 1:
        text_msg = f"Door {nav_instruction}"
    else:
        text_msg = f"{count} doors detected, nearest {nav_instruction}"
    
    # Determine distance for audio type
    distance = estimate_door_distance(main_detection['bbox'], frame_height)
    
    # Use haptic (success type) or beep based on distance
    if use_haptic:
        if distance == "very close":
            audio_type = 'success'  # Haptic feedback for very close
        elif distance == "close":
            audio_type = 'beep_high'  # High beep for close
        else:
            audio_type = 'speech'  # Regular speech for medium/far
    else:
        audio_type = 'speech'
    
    # For haptic/beep, we still want to speak the message
    # The audio type affects additional feedback beyond the spoken text
    
    return {
        'audio': {
            'type': audio_type,
            'text': text_msg,
            'rate': 1.0,
            'interrupt': distance in ['very close', 'close']  # Interrupt for proximity alerts
        },
        'text': text_msg,
        'detections': detections
    }


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """
    Main entry point for door and frame detection tool.
    
    Detects doors and door frames and provides verbal navigation instructions 
    using clock face positions with beep and haptic feedback.
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration:
            - confidence: Detection threshold (default 0.25)
            - is_streaming: Streaming mode flag (default False)
    
    Returns:
        Dictionary with audio configuration and text, or empty string in streaming mode
        when no state change detected.
        
        Dictionary format:
            {
                'audio': {
                    'type': 'success',  # 'success' (haptic), 'beep_high', or 'speech'
                    'text': 'Door straight ahead, move forward',
                    'rate': 1.0,
                    'interrupt': True  # For proximity alerts
                },
                'text': 'Door straight ahead, move forward'
            }
        
        Or empty string "" in streaming mode when no change (prevents repetition).
        
        Audio types by distance:
            - Very close (>60% frame): 'success' (haptic vibration)
            - Close (>30% frame): 'beep_high' (high-pitched beep)
            - Medium/far: 'speech' (normal TTS)
    
    Note:
        Requires mobile app with audio.type support (e.g., github-divergence branch).
    """
    global _last_door_detection, _last_door_count, _last_door_bbox, _frames_since_door, _consecutive_no_door_frames
    
    # Handle None or invalid image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {
                'type': 'error',
                'text': 'No camera image available',
                'rate': 1.0
            },
            'text': 'No camera image available'
        }
    
    # Parse input_data
    config = {}
    if isinstance(input_data, dict):
        config = input_data
    
    # Get configuration parameters
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    is_streaming = config.get('is_streaming', False)
    
    # Get frame dimensions
    height, width = image.shape[:2]
    
    # Detect doors
    detections = detect_doors(image, confidence)
    
    # Proximity inference: if no door detected but one was seen recently,
    # infer the user is very close (door fills frame, YoloWorld can't recognize it)
    if not detections and _last_door_bbox is not None and _frames_since_door < PROXIMITY_FRAME_WINDOW:
        # Infer door is still there but too close to detect
        # Create a synthetic detection at center of frame with full coverage
        # This assumes the door is straight ahead and fills the entire view
        detections = [{
            'bbox': [0, 0, width, height],  # Full frame
            'center': (width // 2, height // 2),  # Center of frame
            'class': 'door (inferred - very close)',
            'confidence': INFERRED_CONFIDENCE
        }]
    
    # Update proximity tracking based on detections
    if detections:
        # Check if this is an inferred detection
        is_inferred = detections[0].get('class') == 'door (inferred - very close)'
        
        # Door detected (real or inferred) - update tracking
        _frames_since_door = 0
        
        # Only reset consecutive no-door frames for REAL detections, not inferred ones
        # Inferred detections are just proximity inference, not actual door sightings
        if not is_inferred:
            _consecutive_no_door_frames = 0  # Reset no-door counter only for real detections
            # Store the largest real detection's bbox for future proximity inference
            largest = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
            _last_door_bbox = largest['bbox']
    else:
        # No door detected - increment counters
        _frames_since_door += 1
        # Only increment consecutive no-door frames if we're OUTSIDE the proximity inference window
        # This prevents announcing "no door" immediately after proximity inference ends
        if _frames_since_door > PROXIMITY_FRAME_WINDOW:
            _consecutive_no_door_frames += 1
        # Reset tracking after threshold
        if _frames_since_door > RESET_FRAME_THRESHOLD:
            _last_door_bbox = None
    
    # Use dictionary format with audio configuration for beep/haptic feedback
    # The github-divergence branch of the mobile app supports audio.type
    
    if not detections:
        # No door detected
        if is_streaming:
            # In streaming mode, use temporal smoothing to prevent transient "no door" messages
            # Only announce "no door" after consecutive frames without detection
            # AND after the proximity inference window has fully elapsed
            if (_last_door_detection != 'no_door' and 
                _consecutive_no_door_frames >= NO_DOOR_ANNOUNCEMENT_DELAY and
                _frames_since_door > PROXIMITY_FRAME_WINDOW):
                _last_door_detection = 'no_door'
                _last_door_count = 0
                # Return dict format with speech type
                return {
                    'audio': {
                        'type': 'speech',
                        'text': 'No door detected',
                        'rate': 1.0
                    },
                    'text': 'No door detected'
                }
            else:
                # Still no door OR not enough consecutive frames yet OR still in proximity window - return empty to avoid repetition/noise
                return ""
        else:
            # One-shot mode: always return
            return {
                'audio': {
                    'type': 'speech',
                    'text': 'No door detected',
                    'rate': 1.0
                },
                'text': 'No door detected'
            }
    
    # Use the most prominent detection (largest)
    main_detection = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
    
    # Generate navigation instruction
    nav_instruction = generate_navigation_instruction(
        main_detection, width, height, is_streaming
    )
    
    # Build text message
    count = len(detections)
    if count == 1:
        text_msg = f"Door {nav_instruction}"
    else:
        text_msg = f"{count} doors detected, nearest {nav_instruction}"
    
    # Determine distance for audio type (beep/haptic feedback)
    distance = estimate_door_distance(main_detection['bbox'], height)
    
    # Select audio type based on distance
    if distance == "very close":
        audio_type = 'success'  # Haptic feedback for very close
        audio_config = {
            'type': audio_type,
            'text': text_msg,
            'rate': 1.0,
            'interrupt': True
        }
    elif distance == "close":
        audio_type = 'beep_high'  # High beep for close
        audio_config = {
            'type': audio_type,
            'text': text_msg,
            'rate': 1.0,
            'interrupt': True,
            'duration': 300,  # 300ms beep duration for better audibility
            'frequency': 880  # High frequency beep
        }
    else:
        audio_type = 'speech'  # Regular speech for medium/far
        audio_config = {
            'type': audio_type,
            'text': text_msg,
            'rate': 1.0,
            'interrupt': False
        }
    
    # Streaming mode: track state to avoid repetition
    if is_streaming:
        # Create a simple state key based on position and count
        clock_pos = get_clock_position(main_detection['center'][0], main_detection['center'][1], width, height)
        current_detection = f"door_at_{clock_pos}"
        
        # Check if anything changed
        if _last_door_detection != current_detection or _last_door_count != count:
            # State changed - announce it
            _last_door_detection = current_detection
            _last_door_count = count
            # Also reset the consecutive no-door frames since we're announcing a door
            # This prevents accumulated no-door frames from causing a spurious announcement
            _consecutive_no_door_frames = 0
            return {
                'audio': audio_config,
                'text': text_msg
            }
        else:
            # Same door in same position - no change, return empty
            # But still reset consecutive no-door frames since we DO have a door
            # This is critical to prevent "no door" from being announced when doors are present
            _consecutive_no_door_frames = 0
            return ""
    else:
        # One-shot mode: always return
        # Reset streaming state when not in streaming mode
        _last_door_detection = None
        _last_door_count = 0
        _last_door_bbox = None
        _frames_since_door = 0
        _consecutive_no_door_frames = 0
        return {
            'audio': audio_config,
            'text': text_msg
        }


# Building block exports for use by other tools
__all__ = [
    'main',
    'detect_doors',
    'get_clock_position',
    'estimate_door_distance',
    'generate_navigation_instruction',
    'create_door_detection_response'
]
