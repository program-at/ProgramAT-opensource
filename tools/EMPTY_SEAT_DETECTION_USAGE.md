# Empty Seat Detection Tool

## Overview
The Empty Seat Detection Tool helps blind or low vision users identify available seating in a room. It detects chairs, determines which are empty versus occupied, and provides audio-friendly guidance about seat locations and navigation.

## Features
- **Chair Detection**: Uses YOLO11 with COCO dataset to detect chairs, couches, and benches
- **Occupancy Detection**: Determines which seats are empty by analyzing proximity to detected people
- **Spatial Awareness**: Groups seats by location (left, right, front, back, center)
- **Navigation Guidance**: Provides directional cues to help users find the nearest empty seat
- **Audio-Optimized Output**: Returns natural language descriptions suitable for text-to-speech

## How It Works
1. The tool receives a camera frame from the mobile app
2. Uses YOLO object detection to identify all chairs and people in the scene
3. Analyzes overlap between people and chairs to determine occupancy
4. Groups empty seats by their spatial location
5. Returns an audio description with count, locations, and navigation guidance

## Example Usage

### Basic Usage
```python
import cv2
from empty_seat_detection import main

# Capture camera frame
image = cv2.imread('room_with_chairs.jpg')

# Run the tool
result = main(image)
print(result)
# Output: "There are 5 empty seats: two on the left, three on the front right. Turn left and walk forward about 10 feet."
```

### With Configuration Options
```python
# Disable navigation guidance
result = main(image, {'include_navigation': False})
# Output: "There are 5 empty seats: two on the left, three on the front right"

# Adjust detection confidence threshold
result = main(image, {'confidence': 0.6})

# Custom occupancy threshold
result = main(image, {'occupancy_threshold': 0.3})
```

## Configuration Options

The tool accepts an optional `input_data` dictionary with the following parameters:

- **confidence** (float, default: 0.5): Detection confidence threshold (0.0 to 1.0)
  - Higher values = fewer but more confident detections
  - Lower values = more detections but potentially more false positives

- **include_navigation** (bool, default: True): Include navigation guidance in output
  - When True: Provides directional cues to nearest empty seat
  - When False: Only reports count and locations

- **occupancy_threshold** (float, default: 0.4): IoU threshold for occupancy detection
  - Determines how much overlap is needed between person and chair to consider it occupied
  - Higher values = more strict (requires more overlap)
  - Lower values = more lenient (less overlap needed)

## Output Examples

### Multiple Empty Seats
```
"There are 5 empty seats: two on the left, three on the front right. Turn left and walk forward about 10 feet."
```

### Single Empty Seat
```
"There is 1 empty seat: one on the center. Continue straight ahead and walk forward a few steps."
```

### No Empty Seats
```
"I see 6 seats, but all appear to be occupied"
```

### No Seats Detected
```
"No seats detected in view. Please scan the room to find seating areas"
```

### No Camera Image
```
"No camera image available"
```

## Technical Details

### Object Detection
- Uses YOLO11 (nano model) for efficient CPU-based detection
- Detects three seat types: chairs, couches, and benches
- Also detects people to determine occupancy

### Occupancy Algorithm
The tool determines if a seat is occupied using two methods:
1. **IoU (Intersection over Union)**: Calculates overlap between person and chair bounding boxes
2. **Center Point Check**: Checks if a person's center point falls within a chair's bounding box

A seat is considered occupied if either condition is met above the threshold.

### Position Descriptions
The frame is divided into a 3x3 grid:
- **Horizontal**: left, center, right
- **Vertical**: back (top), middle, front (bottom)

This provides 9 possible positions: "back left", "back", "back right", "left", "center", "right", "front left", "front", "front right"

### Navigation Guidance
The tool provides simple directional guidance:
1. Identifies the closest empty seat (based on vertical position in frame)
2. Generates horizontal direction: "Turn left", "Turn right", or "Continue straight ahead"
3. Estimates distance based on seat size in frame

## Integration with ProgramAT

This tool follows the standard ProgramAT tool interface:
- Main entry point: `main(image, input_data)`
- Runs on the backend server
- Receives camera frames from mobile app
- Returns audio-friendly strings or dictionaries

The mobile app automatically:
- Sends camera frames to the tool
- Receives the audio description
- Speaks the result via text-to-speech

## Building Blocks

The tool exports several reusable functions for other tools:
- `detect_objects()`: YOLO object detection
- `calculate_iou()`: Intersection over Union calculation
- `is_chair_occupied()`: Occupancy detection
- `get_position_description()`: Spatial position descriptions
- `group_seats_by_location()`: Location-based grouping
- `generate_navigation_guidance()`: Navigation cue generation
- `create_audio_description()`: Audio-friendly description creation

## Testing

Run the test suite:
```bash
cd backend
python test_empty_seat_detection.py
```

The test suite validates:
- IoU calculations
- Occupancy detection logic
- Position descriptions
- Seat grouping by location
- Navigation guidance generation
- Audio description formatting
- Main function integration

## Limitations

1. **Detection Accuracy**: Depends on YOLO model accuracy and scene conditions
2. **Lighting**: Poor lighting may reduce detection quality
3. **Viewing Angle**: Works best with a clear view of the seating area
4. **Seat Types**: Only detects chairs, couches, and benches (from COCO dataset)
5. **Distance Estimation**: Rough approximation based on visual size, not actual measurements

## Future Enhancements

Potential improvements:
- Support for additional seat types
- More precise distance estimation using depth cameras
- Seat reservation/availability integration
- Multi-frame analysis for improved accuracy
- Support for detecting partial occupancy (e.g., shared benches)
