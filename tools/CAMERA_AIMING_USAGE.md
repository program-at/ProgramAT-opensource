# Camera Aiming Assistance Tool

## Overview

The Camera Aiming Assistance tool helps blind users center their camera on an object to take well-framed photos. It provides real-time directional audio cues like "move left", "move right", "move up", "move down", "move closer", and "move further away" to guide users in framing objects properly.

This tool is essential for assistive technology apps where photo quality significantly impacts performance. It uses YOLOv11 and COCO dataset for object detection.

## How It Works

1. **Object Detection**: Uses YOLO11 to detect objects in the camera frame
2. **Object Locking**: Locks onto a target object (largest by default) and tracks it
3. **Framing Analysis**: Calculates how centered and well-sized the object is
4. **Directional Guidance**: Provides audio cues to help user adjust camera position
5. **Completion Detection**: Announces "Well framed!" when object is properly positioned

## Usage

### Basic Usage

The tool runs automatically when invoked by the mobile app. It receives camera frames and returns audio guidance:

```python
from camera_aiming import main
import cv2

# Load camera frame
image = cv2.imread('camera_frame.jpg')

# Get aiming guidance
guidance = main(image)
print(guidance)  # "Move right and back up"
```

### Configuration Options

Pass configuration via `input_data` dictionary:

```python
config = {
    'confidence': 0.5,          # Detection confidence threshold (0.0-1.0)
    'target_class': 'keyboard', # Specific object to aim at (default: auto)
    'centering_tolerance': 0.15,# How centered object must be (0.0-1.0)
    'size_min': 0.15,           # Minimum size ratio for good framing
    'size_max': 0.7,            # Maximum size ratio for good framing
    'lock_object': True,        # Lock onto first detected object
    'reset': False              # Reset object lock
}

guidance = main(image, config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence` | float | 0.5 | Minimum confidence for object detection (0.0-1.0) |
| `target_class` | str | None | Specific COCO class to aim at (e.g., 'keyboard', 'person', 'cup'). If None, auto-selects largest object |
| `centering_tolerance` | float | 0.15 | How centered the object must be (15% tolerance from center) |
| `size_min` | float | 0.15 | Minimum size ratio - object should be at least 15% of frame |
| `size_max` | float | 0.7 | Maximum size ratio - object should be at most 70% of frame |
| `lock_object` | bool | True | Whether to lock onto first detected object and track it |
| `reset` | bool | False | Reset object lock and find new object |

## Example Usage Scenario

**Scenario**: User wants to take a photo of a keyboard

1. **Frame 1**: Keyboard is detected but too close and to the left
   - Output: "Aiming at keyboard. Move right and back up"

2. **Frame 2**: User moved right but still too close
   - Output: "Back up a little"

3. **Frame 3**: User backed up, distance is good but slightly off-center
   - Output: "Move slightly left"

4. **Frame 4**: Keyboard is well-framed
   - Output: "Well framed! Keyboard is centered and ready"

## Framing Criteria

The tool considers an object "well-framed" when:

1. **Centered**: Object center is within ±15% of frame center (configurable)
2. **Good Size**: Object occupies 15-70% of frame (configurable)
3. **Optimal**: Object is close to ideal size (~40% of frame)

## Directional Cues

### Horizontal Positioning
- **"Move left"** - Object is too far to the right (>30% offset)
- **"Move slightly left"** - Object is to the right (15-30% offset)
- **"Move right"** - Object is too far to the left (<-30% offset)
- **"Move slightly right"** - Object is to the left (-30% to -15% offset)

### Vertical Positioning
- **"Move up"** - Object is too far down (>30% offset)
- **"Move slightly up"** - Object is down (15-30% offset)
- **"Move down"** - Object is too far up (<-30% offset)
- **"Move slightly down"** - Object is up (-30% to -15% offset)

### Distance Adjustment
- **"Back up"** - Object is too close (>70% of frame)
- **"Back up a little"** - Object is close (60-70% of frame)
- **"Move closer"** - Object is too far (<15% of frame)
- **"Move a bit closer"** - Object is far (15-20% of frame)

### Ready State
- **"Hold steady"** - Object is well-positioned, hold current position
- **"Well framed! [Object] is centered and ready"** - Perfect framing achieved

## Object Locking and Tracking

The tool uses object locking to maintain focus on a single object across frames:

- **First Frame**: Automatically selects the largest object (or specified class)
- **Subsequent Frames**: Tracks the same object even if it moves
- **Lock Loss**: If the locked object disappears, finds a new object
- **Manual Reset**: Use `reset=True` to unlock and select a new object

This prevents the tool from jumping between different objects as the user moves.

## Building Block Functions

These functions can be used by other tools:

### `calculate_framing_metrics(bbox, frame_width, frame_height)`
Calculate detailed framing metrics for an object bounding box.

**Returns**:
```python
{
    'center_x': 0.5,              # Normalized X position (0-1)
    'center_y': 0.5,              # Normalized Y position (0-1)
    'size_ratio': 0.4,            # Size as ratio of frame
    'horizontal_offset': 0.0,     # Distance from center (-0.5 to 0.5)
    'vertical_offset': 0.0,       # Distance from center (-0.5 to 0.5)
    'is_centered': True,          # Whether centered within tolerance
    'is_good_size': True,         # Whether size is in good range
    'bbox': [x, y, w, h]         # Original bounding box
}
```

### `generate_directional_cues(metrics)`
Generate audio-friendly directional guidance from metrics.

**Example**:
```python
metrics = calculate_framing_metrics([100, 100, 200, 200], 640, 480)
cues = generate_directional_cues(metrics)
# "Move right and move closer"
```

### `is_well_framed(metrics)`
Check if an object is well-framed and ready for photo.

**Example**:
```python
metrics = calculate_framing_metrics(bbox, width, height)
if is_well_framed(metrics):
    print("Perfect framing!")
```

### `select_target_object(detections, target_class, lock_object)`
Select which object to aim at from a list of detections.

**Example**:
```python
from object_recognition import detect_objects

detections = detect_objects(image)
target = select_target_object(detections, target_class='keyboard', lock_object=True)
```

### `reset_lock()`
Reset the object lock state to find a new object.

## Integration with Object Recognition

This tool uses building blocks from `object_recognition.py`:

```python
from object_recognition import detect_objects, COCO_CLASSES

# Detect objects in frame
detections = detect_objects(image, confidence_threshold=0.5)

# Each detection contains:
# - class_name: Object type (from COCO_CLASSES)
# - confidence: Detection confidence (0.0-1.0)
# - bbox: Bounding box [x, y, width, height]
# - center: Center point [x, y]
```

## Audio-Friendly Output

All outputs are designed for text-to-speech:

- Natural language sentences
- Concise and actionable
- No technical jargon or abbreviations
- Clear pronunciation

Examples:
- ✅ "Move right and back up"
- ✅ "Well framed! Keyboard is centered and ready"
- ✅ "Hold steady"
- ❌ "h_offset: 0.35, v_offset: -0.12"
- ❌ "bbox: [120, 80, 300, 250]"

## Supported Object Classes

The tool can detect and aim at any of the 80 COCO object classes:

- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Outdoor**: traffic light, fire hydrant, stop sign, parking meter, bench
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Accessories**: backpack, umbrella, handbag, tie, suitcase
- **Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
- **Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Furniture**: chair, couch, potted plant, bed, dining table, toilet
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone
- **Appliances**: microwave, oven, toaster, sink, refrigerator
- **Indoor**: book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Tips for Best Results

1. **Start with a wider view**: Begin with the camera pointed generally at the object
2. **Follow cues sequentially**: Make one adjustment at a time
3. **Be patient**: Wait for the tool to recognize your adjustments
4. **Use object locking**: Let the tool track the same object across frames
5. **Reset if needed**: If you want to aim at a different object, use `reset=True`

## Error Handling

The tool handles various error conditions gracefully:

- **No objects detected**: "No objects detected, please point camera at an object"
- **Target class not found**: "Cannot find [class], please point camera at one"
- **No camera image**: "No camera image available"
- **Lock reset**: "Lock reset, finding new object"

## Performance Considerations

- Uses YOLOv11n (nano) model for fast CPU inference
- Object locking reduces computation by tracking existing objects
- Minimal processing for quick real-time feedback
- Designed for mobile camera streaming at ~1-5 fps

## Testing

Run the test suite to verify functionality:

```bash
cd backend
python test_camera_aiming.py
```

The test suite includes:
- Framing metrics calculation
- Directional cue generation
- Well-framed detection
- Object locking and tracking
- Main function integration
