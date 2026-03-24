# Door and Frame Detection with Navigation Aid

## Overview

A tool that provides auditory (beep) and haptic (vibration) feedback for door and frame presence, along with verbal navigation instructions using clock face navigation.

## Features

- **Door Detection**: Detects doors and door frames using YoloWorld (since door is not in COCO classes)
- **Haptic Feedback**: Provides vibration feedback for very close doors via 'success' audio type
- **Beep Alerts**: Uses high beep for close doors
- **Clock Face Navigation**: Provides navigation using clock positions (1-3, 9-12 o'clock)
- **Streaming Mode**: Limits responses to 15 words in streaming mode
- **Audio-Optimized**: All outputs are designed for text-to-speech

## Problem It Solves

Helps visually impaired users detect doors and door frames in their environment and navigate toward them safely. The tool provides multi-modal feedback (haptic, beep, speech) to indicate door presence and proximity.

## How It Works

1. **Detection**: Uses YoloWorld model to detect doors and door frames
2. **Distance Assessment**: Calculates relative distance based on bounding box size
3. **Position Mapping**: Maps door position to clock face (1-3, 9-12 only)
4. **Feedback Selection**: 
   - Very close (>60% frame height): Haptic feedback (success type)
   - Close (>30% frame height): High beep
   - Medium/Far: Regular speech
5. **Navigation**: Provides verbal instructions based on position and distance

## Clock Face Navigation

The tool uses clock positions visible from camera perspective:
- **12 o'clock**: Straight ahead (top center)
- **1-3 o'clock**: Right side (top to bottom)
- **9-11 o'clock**: Left side (top to bottom)
- Positions 4-8 are not used (would be behind camera)

## Usage from Mobile App

The tool automatically receives camera frames from the mobile app and returns audio feedback with optional haptic/beep alerts.

### Input Parameters (via input_data)

```python
{
    'confidence': 0.25,       # Detection confidence threshold (default: 0.25)
    'is_streaming': False,    # Enable streaming mode with state tracking
}
```

### Output Format

The tool returns dictionaries with audio configuration for beep and haptic feedback:

```python
{
    'audio': {
        'type': 'success',      # 'success' (haptic), 'beep_high', or 'speech'
        'text': 'Door straight ahead, move forward',
        'rate': 1.0,
        'interrupt': True       # For proximity alerts
    },
    'text': 'Door straight ahead, move forward'
}
```

Or empty string `""` in streaming mode when no state change (prevents repetition).

### Audio Types by Distance

The tool provides distance-based audio feedback:
- **Very close doors** (>60% frame): `success` type (haptic vibration)
- **Close doors** (>30% frame): `beep_high` type (high-pitched beep)
- **Medium/far doors**: `speech` type (normal TTS)

**Note**: Requires mobile app with audio.type support (e.g., github-divergence branch).

## Example Scenarios

### Scenario 1: Approaching a Door
```
User walks toward a door:
- Far: "Door straight ahead, walk ahead" (speech)
- Medium: "Door straight ahead, walk forward" (speech)
- Close: "Door straight ahead, move forward" (high beep + speech)
- Very close: "Door straight ahead, reach forward" (haptic + speech)
```

### Scenario 2: Door to the Right
```
Door at 2 o'clock position:
- "Door at 2 o'clock, walk forward" (speech)
- User adjusts direction
- "Door straight ahead, move forward" (high beep + speech)
```

### Scenario 3: Multiple Doors
```
Multiple doors in hallway:
- "3 doors detected, nearest straight ahead, move forward"
```

## Streaming Mode

In streaming mode (`is_streaming: True`):
- Responses are limited to 15 words maximum
- Returns empty string `""` when no state change detected (prevents repetition)
- Only announces when:
  - Door first detected (after "no door" state)
  - Door disappears (after being detected)
  - Door position changes (different clock position)
  - Number of doors changes
- Same pattern as package_finding.py for smooth streaming experience

## Building Block Functions

The tool exports reusable functions for other tools:

```python
from door_detection import (
    detect_doors,                      # Detect doors using YoloWorld
    get_clock_position,                # Convert position to clock face
    estimate_door_distance,            # Estimate distance to door
    generate_navigation_instruction,   # Create navigation text
    create_door_detection_response     # Build full response dict
)
```

## Dependencies

- **ultralytics**: For YoloWorld model
- **opencv-python**: For image processing
- **numpy**: For array operations

The YoloWorld model (`yolov8s-world.pt`) is automatically downloaded on first use.

### Model Caching for Real-Time Performance

The tool uses a shared `yolo_model_cache` dictionary (injected by the backend server) to cache the YoloWorld model across all tool executions. This is critical for real-time performance:

- **Without caching**: Model reloads on every frame (~2-3 seconds per frame)
- **With caching**: Model loaded once, reused for all frames (~0.1 seconds per frame)

The cache stores both the model and the custom classes to avoid redundant `set_classes()` calls.

## Technical Notes

### Why YoloWorld?

Standard COCO dataset does not include 'door' or 'door frame' classes. YoloWorld enables detection of arbitrary objects by text prompt, making it ideal for detecting doors.

### Clock Position Mapping

The frame is divided into a 3x3 grid:
- Horizontal: left (0-33%), center (33-67%), right (67-100%)
- Vertical: top (0-33%), middle (33-67%), bottom (67-100%)

Clock positions are assigned based on grid region.

### Distance Estimation

Distance is estimated using object height as percentage of frame:
- **Very Close**: >60% of frame height
- **Close**: 30-60% of frame height
- **Medium**: 15-30% of frame height
- **Far**: <15% of frame height

## Testing

Run the test suite:

```bash
cd backend
python test_door_detection.py
```

The test suite validates:
- Clock position calculation
- Distance estimation
- Navigation instruction generation
- Streaming mode word limits
- Audio response type selection
- Error handling

## Limitations

1. Requires YoloWorld model to be downloaded (happens automatically)
2. Detection accuracy depends on lighting and door visibility
3. Works best with standard rectangular doors
4. May not detect very ornate or unusual door designs
5. Requires network connectivity for initial model download

## Future Enhancements

- Add door open/closed state detection
- Detect door handles for precise interaction
- Add doorway width estimation
- Support for sliding doors and other door types
- Distance estimation in actual units (feet/meters)
