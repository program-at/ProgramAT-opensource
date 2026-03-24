# Live Object Recognition Tool

A real-time object detection tool designed specifically for blind and low vision users to identify objects in their environment through audio feedback.

## Overview

This tool uses YOLOv11 with the COCO dataset to detect and describe objects visible in camera frames. It provides natural language descriptions optimized for text-to-speech output.

## Features

- **Real-time Detection**: Identifies 80 different object classes from the COCO dataset
- **Audio-Friendly Output**: Natural language descriptions perfect for text-to-speech
- **Spatial Awareness**: Reports object positions (left, right, center, top, bottom)
- **Distance Estimation**: Estimates relative distance (very close, close, medium, far)
- **Smart Tracking**: In streaming mode, only announces new objects to reduce repetition
- **Fallback Support**: Works even without YOLO using OpenCV face detection

## Detected Object Classes (80 total)

**Common Items**: person, chair, couch, bed, dining table, book, clock, vase, bottle, cup, bowl

**Electronics**: laptop, cell phone, keyboard, mouse, remote, tv, microwave, oven, toaster, refrigerator

**Outdoor/Vehicles**: car, bicycle, motorcycle, bus, train, truck, traffic light, stop sign

**Kitchen**: fork, knife, spoon, bowl, banana, apple, sandwich, pizza, wine glass

**And many more...** (full list in the code)

## Usage

### Basic Usage (Single Frame)

```python
from object_recognition import main
import cv2

# Capture or load an image
image = cv2.imread('kitchen.jpg')

# Run detection with defaults
result = main(image, {})
print(result)
# Output: "I see 5 objects: 2 cups on the left, 1 laptop in the center, 2 chairs mostly on the right"
```

### Streaming Mode (Track New Objects)

```python
# In streaming mode, only new objects are announced
result1 = main(frame1, {'track_mode': True})
# Output: "New: I see 3 objects: 2 cups on the left, 1 laptop in the center"

result2 = main(frame2, {'track_mode': True})
# Output: "" (empty - no new objects)

result3 = main(frame3, {'track_mode': True})
# Output: "New: I see 1 object: 1 person on the right close"
```

### Configuration Options

```python
config = {
    'confidence': 0.5,          # Detection threshold (0.0-1.0, default: 0.5)
    'include_positions': True,  # Include spatial positions (default: True)
    'include_distance': True,   # Include distance estimates (default: True)
    'max_objects': 10,          # Max objects to report (default: 10)
    'track_mode': False         # Enable tracking for streaming (default: False)
}

result = main(image, config)
```

### Example Configurations

**Quick Summary (No Details)**
```python
result = main(image, {
    'include_positions': False,
    'include_distance': False
})
# Output: "I see 3 objects: 2 cups, 1 laptop"
```

**Position Only**
```python
result = main(image, {
    'include_distance': False
})
# Output: "I see 3 objects: 2 cups on the left, 1 laptop in the center"
```

**Distance Only**
```python
result = main(image, {
    'include_positions': False
})
# Output: "I see 3 objects: 2 cups close, 1 laptop medium distance"
```

**Lower Confidence (More Detections)**
```python
result = main(image, {
    'confidence': 0.3  # Lower = more detections, may include false positives
})
```

## Real-World Use Cases

### Scenario 1: Leaving for Work
*User points camera at kitchen table*
```
"I see 3 objects: 1 cup on the left, 1 keys in the center, 1 chair on the right"
```
User picks up keys and continues.

### Scenario 2: Entering Foyer (Streaming)
*User enables track_mode and walks through house*
```
Frame 1: "New: I see 2 objects: 2 shoes at the bottom"
Frame 2: "" (no new objects)
Frame 3: "New: I see 2 objects: 1 door in the center, 1 jacket on the left"
```

### Scenario 3: Navigating a Room
*User slowly pans camera to explore*
```
"I see 6 objects: 1 couch on the left very close, 1 tv in the center medium distance, 
 1 remote on the right close, 2 chairs mostly on the right far away, 1 laptop in the center close"
```

## Integration with Mobile App

The tool automatically integrates with the ProgramAT mobile app backend:

1. The app streams camera frames to the backend
2. The backend calls `main(image, input_data)` for each frame
3. Results are automatically spoken via text-to-speech
4. In track_mode, empty results prevent audio spam

## Building Block Functions

This tool exports reusable functions for other tools:

```python
from object_recognition import (
    detect_objects,              # Core detection
    count_objects_by_class,      # Grouping/counting
    get_position_description,    # Spatial positioning
    estimate_distance,           # Distance estimation
    create_audio_description,    # Natural language generation
    track_new_objects            # Object tracking
)

# Example: Use detection in another tool
detections = detect_objects(image, confidence_threshold=0.6)
for det in detections:
    print(f"Found {det['class_name']} at {det['center']}")
```

## Requirements

- **Python 3.8+**
- **OpenCV** (opencv-python): Image processing
- **NumPy**: Array operations
- **Ultralytics** (optional): YOLOv11 support for better accuracy
  - Falls back to OpenCV Haar Cascade if not available

Install dependencies:
```bash
pip install opencv-python numpy ultralytics
```

## Performance Notes

- **YOLOv11n** (nano): ~200-500ms per frame on CPU, very accurate
- **OpenCV Fallback**: ~50-100ms per frame, only detects faces
- **Recommended Frame Rate**: 1-2 FPS for streaming to balance accuracy and responsiveness

## Troubleshooting

**Issue: "No module named 'ultralytics'"**
- Solution: Install ultralytics (`pip install ultralytics`) or use fallback mode

**Issue: Only detecting "person" class**
- Cause: YOLOv11 not installed, using Haar Cascade fallback
- Solution: Install ultralytics for full 80-class detection

**Issue: Too many objects reported**
- Solution: Increase confidence threshold or reduce max_objects
  ```python
  result = main(image, {'confidence': 0.7, 'max_objects': 5})
  ```

**Issue: Missing small objects**
- Solution: Lower confidence threshold
  ```python
  result = main(image, {'confidence': 0.3})
  ```

## Technical Details

### Detection Pipeline
1. Image received (BGR format from OpenCV)
2. YOLO inference (or fallback to Haar Cascade)
3. Filter by confidence threshold
4. Calculate positions and distances
5. Generate natural language description
6. Return audio-friendly string

### Position Calculation
Frame divided into 3x3 grid:
- Horizontal: left / center / right
- Vertical: top / middle / bottom
- Combined: "top left", "center", "bottom right", etc.

### Distance Estimation
Based on object height relative to frame:
- **Very close**: >60% of frame height
- **Close**: 30-60% of frame height
- **Medium**: 15-30% of frame height
- **Far**: <15% of frame height

### Object Tracking
Objects tracked by class + approximate position (50px buckets):
- Reduces jitter from small movements
- Only reports truly new objects
- Resets when switching out of track_mode

## Future Enhancements

Potential improvements:
- [ ] GPU acceleration for faster inference
- [ ] Custom object classes beyond COCO
- [ ] Object relationship detection ("cup on table")
- [ ] Depth estimation using dual cameras
- [ ] Voice-controlled confidence adjustment

## Support

For issues or questions, please open an issue on the GitHub repository.
