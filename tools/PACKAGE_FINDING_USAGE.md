# Package Finding Assistance Tool

## Overview

A tool that helps blind users identify whether there is a package visible in the camera view. Uses **YoloWorld** for package detection since "package", "box", and "carton" are NOT in the COCO classes.

## Features

- **Still Frame Mode**: Provides yes/no answer with package description and location
- **Streaming Mode**: Provides directional guidance to help locate packages
- **Audio-Friendly Output**: All responses are optimized for text-to-speech
- **Distance Estimation**: Estimates how far away the package is
- **Position Description**: Describes where the package is located in the frame

## Detection Technology

This tool uses **YoloWorld** (not YOLO11 with COCO) because:
- "package" is NOT in the COCO classes
- "box" is NOT in the COCO classes  
- "carton" is NOT in the COCO classes

YoloWorld supports open-vocabulary detection, allowing us to detect custom classes like "package", "box", "carton", and "parcel".

## Usage Examples

### Still Frame Mode (Default)

**Scenario**: User takes a photo of their front porch

**Input**: 
- Image: Photo from camera
- Mode: 'still' (default)

**Output Examples**:
- `"Yes, package found! Medium sized box near the left side corner"`
- `"Yes, 2 packages found! Largest is a large box near the center"`
- `"No package detected"`

### Streaming Mode

**Scenario**: User walks outside streaming camera feed

**Input**:
- Image: Live camera stream
- Mode: 'stream'

**Output Examples**:
1. Initially (no package): `"No package found"`
2. Package comes into view: `"Package detected! Medium box. Move forward, package straight ahead, about 5 to 10 feet away"`
3. Getting closer: `"Package is straight ahead, about 2 to 3 feet away"`
4. Very close: `"Package is straight ahead, very close, within reach"`

## Configuration Options

Configure the tool via the `input_data` parameter:

```python
input_data = {
    'mode': 'still',  # or 'stream'
    'confidence': 0.4,  # Detection threshold (0.0 to 1.0)
    'package_classes': ['package', 'box', 'carton', 'parcel']  # Classes to detect
}
```

### Parameters

- **mode**: `'still'` or `'stream'`
  - `'still'`: Single frame analysis with yes/no response
  - `'stream'`: Continuous monitoring with directional guidance
  - Default: `'still'`

- **confidence**: Detection confidence threshold (0.0 to 1.0)
  - Lower values detect more packages but may have false positives
  - Higher values are more selective but may miss packages
  - Default: `0.4` (optimized for YoloWorld)

- **package_classes**: List of object types to detect
  - Default: `['package', 'box', 'carton', 'parcel']`
  - Can be customized for specific use cases

## From Mobile App

The tool integrates automatically with the ProgramAT mobile app:

1. User selects "Package Finding" tool
2. Camera frames are sent to backend
3. Tool processes frames and returns audio feedback
4. Audio is played via text-to-speech on device

## Technical Details

### YoloWorld Model

- Model: `yolov8s-world.pt`
- Supports open-vocabulary detection
- Custom class setting via `model.set_classes()`
- Downloads automatically on first use
- **Performance optimization**: Model is cached globally and loaded only once, enabling real-time detection at YOLO speeds

### Model Caching

The tool implements model caching for optimal performance:
- First frame: Model loads (~1-2 seconds one-time delay)
- Subsequent frames: Instant inference (true real-time performance)
- Model persists in memory across all tool executions via backend server cache
- No reload overhead between frames or between different tool runs
- Cache is shared across streaming and one-shot modes

### Building Blocks

The tool reuses functions from other tools:
- `get_position_description()` - Describes object position in frame
- `estimate_distance()` - Estimates distance based on object size
- `get_size_description()` - Describes package size (small/medium/large)

### Streaming Behavior

In streaming mode:
- Returns empty string `""` when no change (prevents audio spam)
- Announces new detections immediately
- Updates when distance category changes
- Provides continuous directional guidance

## Example Code

### Basic Usage (Still Frame)

```python
import cv2
from package_finding import main

# Capture image
image = cv2.imread('front_porch.jpg')

# Detect packages
result = main(image)
print(result)
# Output: "Yes, package found! Medium sized box near the left side corner"
```

### Streaming Mode

```python
import cv2
from package_finding import main

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame in streaming mode
    result = main(frame, {'mode': 'stream'})
    
    # Only speak if there's a result (avoid silence spam)
    if result:
        print(result)  # Would be sent to TTS
```

### Custom Package Classes

```python
# Look for specific types
result = main(image, {
    'mode': 'still',
    'package_classes': ['cardboard box', 'envelope', 'package']
})
```

## Limitations

- Requires good lighting for accurate detection
- May not detect packages in unusual packaging
- Distance estimation is relative, not absolute measurements
- Performance depends on package visibility in frame

## Dependencies

- `ultralytics` - For YoloWorld model
- `opencv-python` - For image processing
- `numpy` - For array operations

Dependencies are auto-installed by the backend module manager.
