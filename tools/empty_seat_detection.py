"""
Empty Seat Detection Tool for Visually Impaired Users

Helps blind or low vision users identify empty seats in a room by detecting chairs
and determining which ones are occupied vs empty.

Features:
- Detects chairs using YOLO11 and COCO dataset
- Identifies occupied vs empty seats by detecting nearby people
- Reports count and general location of empty seats
- Provides navigation guidance to help users find seating
- Audio-optimized output for text-to-speech

Example Usage:
User enters a room and activates the tool. The tool scans and reports:
"There are 5 empty seats. Two are to your left, three are in the front right."
Then provides directional cues like "Turn slightly left and walk forward."

This tool runs on the backend server and receives camera frames from the mobile app.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Detection constants
DEFAULT_CONFIDENCE = 0.5
CHAIR_CLASS = 'chair'
PERSON_CLASS = 'person'
COUCH_CLASS = 'couch'
BENCH_CLASS = 'bench'

# Spatial constants for determining if a chair is occupied
OCCUPANCY_THRESHOLD = 0.4  # Person bbox must overlap at least 40% with chair to be considered occupied

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


def detect_objects(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """
    Detect objects in an image using YOLO11 and COCO dataset.
    
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
        
        # Load YOLO11 model (will auto-download on first use)
        # Using YOLO11n (nano) for speed on CPU
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
        # Fallback if YOLO not available
        pass
    except Exception as e:
        # Log error but don't crash
        pass
    
    return detections


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x, y, width, height]
        bbox2: Second bounding box [x, y, width, height]
        
    Returns:
        IoU value between 0.0 and 1.0
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou


def is_chair_occupied(chair: Dict[str, Any], people: List[Dict[str, Any]]) -> bool:
    """
    Determine if a chair is occupied by checking for nearby people.
    
    A chair is considered occupied if a person's bounding box overlaps
    sufficiently with the chair's bounding box.
    
    Args:
        chair: Chair detection dictionary
        people: List of person detection dictionaries
        
    Returns:
        True if chair is occupied, False if empty
    """
    chair_bbox = chair['bbox']
    
    for person in people:
        person_bbox = person['bbox']
        
        # Calculate overlap
        iou = calculate_iou(chair_bbox, person_bbox)
        
        # If sufficient overlap, chair is occupied
        if iou > OCCUPANCY_THRESHOLD:
            return True
        
        # Also check if person center is inside chair bbox (someone sitting)
        px, py = person['center']
        cx, cy, cw, ch = chair_bbox
        
        if cx <= px <= cx + cw and cy <= py <= cy + ch:
            return True
    
    return False


def get_position_description(center: List[int], width: int, height: int) -> str:
    """
    Get a human-readable position description for an object.
    
    Args:
        center: [x, y] center coordinates of the object
        width: Frame width
        height: Frame height
        
    Returns:
        Position string like "center", "left", "front right", etc.
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
    
    # Determine vertical position (front/back)
    # Bottom of image is closer (front), top is farther (back)
    if y < third_h:
        v_pos = "back"
    elif y < 2 * third_h:
        v_pos = "middle"
    else:
        v_pos = "front"
    
    # Combine positions
    if h_pos == "center" and v_pos == "middle":
        return "center"
    elif h_pos == "center":
        return v_pos
    elif v_pos == "middle":
        return h_pos
    else:
        return f"{v_pos} {h_pos}"


def group_seats_by_location(
    empty_seats: List[Dict[str, Any]], 
    width: int, 
    height: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group empty seats by their location in the frame.
    
    Args:
        empty_seats: List of empty seat detections
        width: Frame width
        height: Frame height
        
    Returns:
        Dictionary mapping location strings to lists of seats in that location
    """
    grouped = defaultdict(list)
    
    for seat in empty_seats:
        location = get_position_description(seat['center'], width, height)
        grouped[location].append(seat)
    
    return dict(grouped)


def generate_navigation_guidance(
    empty_seats: List[Dict[str, Any]],
    grouped_seats: Dict[str, List[Dict[str, Any]]],
    width: int,
    height: int
) -> str:
    """
    Generate navigation guidance to help user find empty seats.
    
    Args:
        empty_seats: List of all empty seats
        grouped_seats: Seats grouped by location
        width: Frame width
        height: Frame height
        
    Returns:
        Navigation guidance string
    """
    if not empty_seats:
        return ""
    
    # Find closest seat (based on y-coordinate - lower y means farther, higher y means closer)
    closest_seat = max(empty_seats, key=lambda s: s['center'][1])
    closest_location = get_position_description(closest_seat['center'], width, height)
    
    # Generate guidance based on closest seat location
    guidance_parts = []
    
    # Horizontal direction
    cx = closest_seat['center'][0]
    third_w = width / 3
    
    if cx < third_w:
        guidance_parts.append("Turn left")
    elif cx > 2 * third_w:
        guidance_parts.append("Turn right")
    else:
        guidance_parts.append("Continue straight ahead")
    
    # Distance estimate (based on size)
    _, _, w, h = closest_seat['bbox']
    seat_size = w * h
    frame_size = width * height
    size_ratio = seat_size / frame_size
    
    if size_ratio > 0.1:
        distance_desc = "a few steps"
    elif size_ratio > 0.05:
        distance_desc = "about 10 feet"
    else:
        distance_desc = "across the room"
    
    guidance_parts.append(f"and walk forward {distance_desc}")
    
    return " ".join(guidance_parts)


def create_audio_description(
    total_seats: int,
    occupied_seats: int,
    empty_seats: List[Dict[str, Any]],
    grouped_seats: Dict[str, List[Dict[str, Any]]],
    width: int,
    height: int,
    include_navigation: bool = True
) -> str:
    """
    Create a natural language audio description of seat availability.
    
    Args:
        total_seats: Total number of seats detected
        occupied_seats: Number of occupied seats
        empty_seats: List of empty seat detections
        grouped_seats: Empty seats grouped by location
        width: Frame width
        height: Frame height
        include_navigation: Whether to include navigation guidance
        
    Returns:
        Audio-friendly description string
    """
    empty_count = len(empty_seats)
    
    if total_seats == 0:
        return "No seats detected in view"
    
    if empty_count == 0:
        return f"I see {total_seats} seat{'s' if total_seats > 1 else ''}, but all appear to be occupied"
    
    # Start description
    if empty_count == 1:
        desc_parts = ["There is 1 empty seat"]
    else:
        desc_parts = [f"There are {empty_count} empty seats"]
    
    # Add location information
    if grouped_seats:
        locations = []
        for location, seats in sorted(grouped_seats.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(seats)
            if count == 1:
                locations.append(f"one on the {location}")
            else:
                locations.append(f"{count} on the {location}")
        
        if locations:
            if len(locations) == 1:
                desc_parts.append(": " + locations[0])
            elif len(locations) == 2:
                desc_parts.append(": " + " and ".join(locations))
            else:
                desc_parts.append(": " + ", ".join(locations[:-1]) + ", and " + locations[-1])
    
    description = "".join(desc_parts)
    
    # Add navigation guidance
    if include_navigation and empty_seats:
        navigation = generate_navigation_guidance(empty_seats, grouped_seats, width, height)
        if navigation:
            description += ". " + navigation
    
    return description


def main(image: np.ndarray, input_data: Any = None) -> str:
    """
    Main entry point for empty seat detection tool.
    
    Detects chairs in the scene, determines which are empty, and provides
    audio-friendly guidance about seat availability and location.
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration:
            - confidence: Detection threshold (default 0.5)
            - include_navigation: Include navigation guidance (default True)
            - occupancy_threshold: IoU threshold for occupancy detection (default 0.4)
    
    Returns:
        Audio-friendly description string suitable for text-to-speech.
        Example: "There are 5 empty seats: two on the left, three on the front right. Turn left and walk forward about 10 feet."
    """
    # Handle None or invalid image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"
    
    # Parse input_data
    config = {}
    if isinstance(input_data, dict):
        config = input_data
    
    # Get configuration parameters
    confidence = config.get('confidence', DEFAULT_CONFIDENCE)
    include_navigation = config.get('include_navigation', True)
    
    # Get frame dimensions
    height, width = image.shape[:2]
    
    # Detect all objects
    detections = detect_objects(image, confidence)
    
    if not detections:
        return "No objects detected. Please ensure the camera is pointed at the room with seating"
    
    # Filter for seats (chairs, couches, benches)
    seat_classes = {CHAIR_CLASS, COUCH_CLASS, BENCH_CLASS}
    seats = [d for d in detections if d['class_name'] in seat_classes]
    
    # Filter for people
    people = [d for d in detections if d['class_name'] == PERSON_CLASS]
    
    if not seats:
        return "No seats detected in view. Please scan the room to find seating areas"
    
    # Determine which seats are empty
    empty_seats = []
    occupied_seats = []
    
    for seat in seats:
        if is_chair_occupied(seat, people):
            occupied_seats.append(seat)
        else:
            empty_seats.append(seat)
    
    # Group empty seats by location
    grouped_seats = group_seats_by_location(empty_seats, width, height)
    
    # Create audio description
    description = create_audio_description(
        total_seats=len(seats),
        occupied_seats=len(occupied_seats),
        empty_seats=empty_seats,
        grouped_seats=grouped_seats,
        width=width,
        height=height,
        include_navigation=include_navigation
    )
    
    return description


# Building block exports for use by other tools
__all__ = [
    'main',
    'detect_objects',
    'is_chair_occupied',
    'get_position_description',
    'group_seats_by_location',
    'generate_navigation_guidance',
    'create_audio_description',
    'calculate_iou'
]
