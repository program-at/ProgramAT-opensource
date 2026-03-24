# Scene Description Tool

AI-powered scene description tool that generates detailed audio descriptions of images for blind or low vision users.

## Overview

The scene description tool uses Google Gemini's vision capabilities to analyze camera frames and create natural, comprehensive descriptions suitable for text-to-speech output. It's designed with a modular "building block" approach, making it easy to extend and reuse components in other tools.

## Features

- **AI-Powered Analysis**: Uses Google Gemini vision model to understand and describe scenes
- **Audio-Optimized Output**: Generates natural language descriptions perfect for text-to-speech
- **Flexible Configuration**: Multiple detail levels, focus areas, and output styles
- **Building Block Architecture**: Exportable functions for use by other tools
- **Error Handling**: Graceful fallback with helpful error messages
- **Efficient Processing**: Automatic image resizing for optimal API usage

## Usage

### From Mobile App

The tool automatically receives camera frames from the mobile app through the backend WebSocket server. No special setup required - just ensure the GEMINI_API_KEY is configured on the backend.

### Configuration

Configure the tool by sending `input_data` parameters:

```json
{
  "detail_level": "standard",
  "focus": "general",
  "style": "narrative"
}
```

#### Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `detail_level` | `brief`, `standard`, `detailed` | `standard` | How much detail to include |
| `focus` | `general`, `people`, `objects`, `text`, `navigation` | `general` | What to emphasize |
| `style` | `narrative`, `concise` | `narrative` | Audio presentation style |
| `api_key` | string | from env | Optional: Override API key |
| `model` | string | `gemini-3-flash-preview` | Optional: Gemini model to use |

### Detail Levels

- **brief**: 1-2 sentence overview of main elements
- **standard**: Clear description of scene, key objects, people, and activities
- **detailed**: Comprehensive description including spatial relationships, colors, textures, and visible text

### Focus Areas

- **general**: Balanced description of all scene elements
- **people**: Emphasizes people, their positions, appearances, and activities
- **objects**: Focuses on identifying and describing objects and their relationships
- **text**: Carefully identifies and reads any visible text, signs, or labels
- **navigation**: Describes environment for navigation (obstacles, pathways, directions)

### Output Styles

- **narrative**: Natural, flowing descriptions (default)
- **concise**: Direct, bullet-point style information

## Example Outputs

### Brief + General
```
"A sunny park with people sitting on benches near a fountain."
```

### Standard + People
```
"Three people are seated on a wooden park bench. Two adults are on the left side, engaged in conversation, while a child sits on the right side holding what appears to be an ice cream cone. All three are facing toward a central fountain in the background."
```

### Detailed + Navigation
```
"The scene shows a park pathway running diagonally from bottom left to top right. On the left side, approximately 10 feet from the path, stands a wooden bench occupied by three people. The path itself is clear of obstacles, about 4 feet wide, made of gray concrete. A decorative fountain is visible 20 feet ahead in the center of the path. Tree branches overhang the right side of the path at head height."
```

## Building Block Functions

The tool exports several functions that can be used by other tools:

### `analyze_scene(image, api_key, detail_level, focus, model_name)`

Core function that performs AI-powered scene analysis.

```python
from scene_description import analyze_scene

result = analyze_scene(
    image=camera_frame,
    api_key='your-api-key',
    detail_level='detailed',
    focus='navigation'
)

if result['success']:
    print(result['description'])
    print(f"Confidence: {result['confidence']}")
```

### `get_scene_context(image)`

Extract basic context without AI (lighting, colors, dimensions).

```python
from scene_description import get_scene_context

context = get_scene_context(image)
print(f"Lighting: {context['lighting']}")
print(f"Brightness: {context['brightness']}")
print(f"Dominant color: {context['dominant_color']}")
```

### `format_description_for_audio(description, style, max_length)`

Format text for optimal audio presentation.

```python
from scene_description import format_description_for_audio

formatted = format_description_for_audio(
    description=long_text,
    style='concise',
    max_length=200
)
```

### `build_scene_prompt(detail_level, focus, context)`

Build customized prompts for scene description.

```python
from scene_description import build_scene_prompt

prompt = build_scene_prompt(
    detail_level='detailed',
    focus='people',
    context={'lighting': 'dim'}
)
```

### `resize_image_if_needed(image, max_size)`

Resize images efficiently while maintaining aspect ratio.

```python
from scene_description import resize_image_if_needed

resized = resize_image_if_needed(large_image, max_size=(1024, 1024))
```

### `convert_cv2_to_pil(image)`

Convert OpenCV BGR images to PIL RGB format.

```python
from scene_description import convert_cv2_to_pil

pil_image = convert_cv2_to_pil(opencv_image)
```

## Integration Example

### Using as a Standalone Tool

The mobile app can invoke this tool directly via the backend:

1. User points camera at a scene
2. App sends frame to backend with tool name "scene_description"
3. Backend executes the tool with the frame
4. Tool returns audio-friendly description
5. App speaks the description via TTS

### Using as a Building Block

Other tools can import and use scene description functions:

```python
from scene_description import analyze_scene, get_scene_context

def my_custom_tool(image, input_data):
    # Get quick context without AI
    context = get_scene_context(image)
    
    if context['lighting'] == 'very dark':
        return "It's too dark to see clearly. Try using a light."
    
    # Get full AI description
    result = analyze_scene(
        image=image,
        api_key=input_data.get('api_key'),
        detail_level='brief',
        focus='objects'
    )
    
    if result['success']:
        return f"Scene: {result['description']}"
    else:
        return "Unable to analyze scene."
```

## Setup

### Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Pillow (`pillow`)
- Google Generative AI (`google-generativeai`)

All dependencies are automatically installed by the backend's module manager.

### Environment Variables

Set the Gemini API key on the backend server:

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

## Testing

Run the test suite:

```bash
cd backend
python test_scene_description.py
```

The tests verify:
- Error handling without API key
- Building block functions
- Integration with Gemini API (if key available)
- Graceful handling of missing images

## Performance

- **Image Processing**: Automatic resizing to max 1024x1024 for efficiency
- **API Calls**: Uses Gemini 2.0 Flash (fast inference, low latency)
- **Response Time**: Typically 1-3 seconds depending on detail level
- **Cost**: Minimal - Gemini 2.0 Flash is optimized for speed and cost

## Error Handling

The tool handles errors gracefully:

- **No API Key**: Returns helpful error message with setup instructions
- **No Image**: Returns error with audio-friendly explanation
- **API Failure**: Returns error message with details
- **Network Issues**: Timeout handling with clear error messages

All errors return in the standard audio format with `type: 'error'`.

## Accessibility Features

- **Natural Language**: Descriptions sound natural when spoken aloud
- **Context Awareness**: Adapts to lighting conditions and image quality
- **Configurable Detail**: Users can choose how much information they want
- **Focus Modes**: Specialized descriptions for different needs
- **Audio-First Design**: Output optimized for text-to-speech, not visual display

## Future Enhancements

Potential improvements for future versions:

- **Caching**: Cache descriptions for similar scenes to reduce API calls
- **Incremental Updates**: Describe only what changed since last frame
- **Language Support**: Multi-language descriptions
- **Custom Prompts**: User-defined prompt templates
- **Offline Mode**: Basic descriptions without AI when offline
- **Object Tracking**: Track specific objects across frames

## Similar Tools

This tool can be combined with:

- **camera_aiming_assistant**: First aim the camera, then get scene description
- **Text recognition tools**: Get detailed text reading combined with scene context
- **Navigation tools**: Use scene descriptions to inform navigation guidance

## License

Part of the ProgramAT project. See repository LICENSE for details.

## Support

For issues or questions:
- Open an issue on GitHub: https://github.com/program-at/ProgramAT
- Check existing issues for solutions
- Review test suite for usage examples
