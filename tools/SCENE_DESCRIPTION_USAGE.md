# Scene Description Tool - Usage Examples

## Quick Start

The scene description tool is automatically available in the ProgramAT backend. Just ensure the GEMINI_API_KEY is set.

### Setting up the Backend

```bash
cd backend
export GEMINI_API_KEY="your-gemini-api-key-here"
python stream_server.py
```

Get your API key from: https://makersuite.google.com/app/apikey

## Using from Mobile App

The mobile app communicates with the tool via WebSocket messages. Here's how it works:

### 1. One-Shot Description (Single Frame)

Send a message to execute the tool on a single frame:

```json
{
  "type": "execute_tool",
  "tool_name": "scene_description",
  "data": {
    "base64Image": "...",
    "input_data": {
      "detail_level": "standard",
      "focus": "general"
    }
  }
}
```

The backend responds with:

```json
{
  "type": "tool_result",
  "tool_name": "scene_description",
  "result": {
    "success": true,
    "description": "A sunny park with three people sitting on a bench...",
    "confidence": 0.85,
    "audio": {
      "type": "speech",
      "text": "A sunny park with three people sitting on a bench...",
      "rate": 1.0,
      "interrupt": false
    },
    "text": "A sunny park with three people sitting on a bench..."
  }
}
```

The mobile app then speaks the `audio.text` using text-to-speech.

### 2. Streaming Mode (Continuous Descriptions)

For continuous scene analysis, enable streaming mode:

```json
{
  "type": "start_streaming_tool",
  "tool_name": "scene_description",
  "config": {
    "throttle_ms": 5000,
    "input_data": {
      "detail_level": "brief",
      "focus": "navigation",
      "style": "concise"
    }
  }
}
```

The tool will analyze frames every 5 seconds and send updates:

```json
{
  "type": "streaming_tool_result",
  "tool_name": "scene_description",
  "result": {
    "description": "Clear path ahead, trees on both sides.",
    "audio": {
      "type": "speech",
      "text": "Clear path ahead, trees on both sides."
    }
  }
}
```

Stop streaming with:

```json
{
  "type": "stop_streaming_tool",
  "tool_name": "scene_description"
}
```

## Configuration Examples

### For Navigation

Best for blind users navigating spaces:

```json
{
  "detail_level": "detailed",
  "focus": "navigation",
  "style": "concise"
}
```

Output: "Hallway extends 20 feet ahead. Door on left at 10 feet. Stairs begin 15 feet ahead on right."

### For Reading Text

When user needs to read signs or documents:

```json
{
  "detail_level": "detailed",
  "focus": "text",
  "style": "narrative"
}
```

Output: "The sign reads 'Emergency Exit' in red letters. Below it, smaller text says 'Keep door closed at all times.'"

### For Understanding People

When user wants to know about people in the scene:

```json
{
  "detail_level": "standard",
  "focus": "people",
  "style": "narrative"
}
```

Output: "Two people are standing near the entrance. A woman on the left is wearing a blue jacket, and a man on the right is carrying a briefcase."

### For Quick Overview

When user just needs a brief summary:

```json
{
  "detail_level": "brief",
  "focus": "general",
  "style": "concise"
}
```

Output: "Office lobby with reception desk and two people."

## Voice Commands (Future Enhancement)

Potential voice commands that could trigger the tool:

- "Describe what you see"
- "What's in front of me?"
- "Read the scene"
- "Tell me about the people here"
- "Describe the room"
- "What does this say?" (with focus: "text")
- "Where can I walk?" (with focus: "navigation")

## Combining with Other Tools

### Example: Aiming + Description

1. Use `camera_aiming_assistant` to center the subject
2. Once centered, use `scene_description` to get details

```python
# In a custom tool
from camera_aiming_assistant import analyze_frame_for_aiming
from scene_description import analyze_scene

def smart_photo_helper(image, input_data):
    # First, check if properly aimed
    aiming = analyze_frame_for_aiming(image, input_data)
    
    if not aiming['centered']:
        # Return aiming guidance
        return aiming
    
    # Once centered, describe the scene
    description = analyze_scene(
        image=image,
        api_key=input_data.get('api_key'),
        detail_level='standard',
        focus='general'
    )
    
    if description['success']:
        return {
            'audio': {
                'type': 'speech',
                'text': f"Perfect! Ready to shoot. {description['description']}",
                'rate': 1.0
            }
        }
```

## Testing Locally

Test the tool directly without the mobile app:

```python
import sys
import cv2
sys.path.insert(0, 'tools')

import scene_description

# Load a test image
image = cv2.imread('test_photo.jpg')

# Get description
result = scene_description.main(image, {
    'detail_level': 'standard',
    'focus': 'general',
    'api_key': 'your-key-here'
})

print(result['description'])
```

## Performance Tips

1. **Detail Level**: Use 'brief' for faster responses, 'detailed' when accuracy is critical
2. **Streaming Throttle**: Set to 3000-5000ms to balance responsiveness and API costs
3. **Image Size**: Large images are automatically resized to 1024x1024 for efficiency
4. **Focus Areas**: Specific focus areas give better results than 'general'

## Troubleshooting

### No Description Returned

Check that:
- GEMINI_API_KEY is set correctly
- Image is valid (not None or empty)
- Network connection is working

### Generic Descriptions

Try:
- Increasing detail_level to 'detailed'
- Setting a specific focus area
- Ensuring good lighting in the image

### Slow Response

- Use 'brief' detail level
- Reduce image resolution before sending
- Check network latency

## Cost Estimation

Gemini 2.0 Flash pricing (as of 2024):
- Input: ~$0.075 per 1M tokens
- Images: ~0.25 tokens per pixel (resized to max 1024x1024)
- Each description: approximately $0.0001 - $0.0003

For typical usage:
- 100 descriptions/day = ~$0.01 - $0.03/day
- Very cost-effective for personal use

## API Key Security

**Important**: Never commit API keys to git!

- Set via environment variable: `export GEMINI_API_KEY="..."`
- Or use GCP Secret Manager for production
- Backend handles API key, not the mobile app

## Privacy

- Images are sent to Google's Gemini API for analysis
- Images are not stored by Google (per Google's API policy)
- ProgramAT backend does not permanently store images unless `SAVE_FRAMES=True`
- Users should be aware images are processed by third-party AI service

## Next Steps

1. Try the tool with various scene types
2. Experiment with different configurations
3. Combine with other tools for enhanced functionality
4. Provide feedback for improvements

## Support

For issues or questions:
- GitHub Issues: https://github.com/program-at/ProgramAT/issues
- Documentation: tools/SCENE_DESCRIPTION.md
- Tests: backend/test_scene_description.py
