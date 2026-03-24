# Clothing Recognition Tool - Usage Guide

## Overview
The Clothing Recognition Tool identifies the most prominent clothing item in an image and provides a concise description for blind or low vision users. It uses Google Gemini Vision API to provide accurate, natural language descriptions limited to 15 words for quick audio feedback.

## Features
- **Single Item Focus**: Analyzes only the biggest/most prominent clothing item in the image
- **Concise Descriptions**: Limited to 15 words for quick, efficient audio output
- **Category Recognition**: Identifies types of clothing (shirts, pants, shoes, dresses, jackets, etc.)
- **Color Detection**: Describes colors (e.g., "red", "navy blue", "gray")
- **Pattern Recognition**: Identifies patterns (solid, striped, checked, floral, graphic print, etc.)
- **Style Details**: Notes key features like sleeves, necklines, buttons, logos
- **Audio-Optimized Output**: Natural language descriptions perfect for text-to-speech

## Requirements
- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- Google Generative AI (`google-generativeai`)
- Gemini API Key (set as `GEMINI_API_KEY` environment variable)

## Installation
```bash
pip install opencv-python numpy pillow google-generativeai
```

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage
```python
from clothing_recognition import main
import cv2

# Load an image
image = cv2.imread('clothing_photo.jpg')

# Analyze clothing (returns dict with 'description' key)
result = main(image)
print(result['description'])
# Output: "Red t-shirt with graphic print."
```

### With Custom Parameters
```python
# Brief description (still under 15 words)
result = main(image, {'detail_level': 'brief'})
print(result['description'])
# Output: "Red t-shirt, graphic print."

# Detailed description (still under 15 words, but more features)
result = main(image, {'detail_level': 'detailed'})
print(result['description'])
# Output: "Bright red cotton t-shirt, short sleeves, crew neck, cartoon logo."

# Standard description (default, under 15 words)
result = main(image, {'detail_level': 'standard'})
print(result['description'])
# Output: "Red t-shirt with graphic print, cartoon character logo."
```

**Note**: All descriptions are limited to 15 words maximum for concise audio output. The tool focuses only on the most prominent clothing item in the image.

### Configuration Options
The tool accepts an optional `input_data` dictionary with the following parameters:

- **detail_level**: Controls description detail (all under 15 words)
  - `'brief'`: Very concise (e.g., "Red shirt, graphic print")
  - `'standard'`: Balanced description with key features (default)
  - `'detailed'`: More features included, still under 15 words

- **api_key**: Optional Gemini API key override (uses environment variable by default)

- **model**: Optional Gemini model override (default: `'gemini-3-flash-preview'`)

## Example Outputs

**All outputs are limited to 15 words maximum and focus only on the most prominent clothing item.**

### Single Item Examples
**Input**: Photo of a blue denim jacket

**Output**: 
```
"Blue denim jacket, button front, two chest pockets, medium wash."
```

**Input**: Photo of a red t-shirt

**Output**:
```
"Red t-shirt with graphic print."
```

**Input**: Photo of striped pants

**Output**:
```
"Navy and white striped pants, horizontal stripes."
```

**Note**: When multiple clothing items are present, the tool describes only the largest or most prominent one.

## Error Handling
The tool handles errors gracefully:

```python
# No image provided
result = main(None)
# Output: "No camera image available"

# API key not configured
result = main(image)  # Without GEMINI_API_KEY set
# Output: "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
```

## Building Block Functions
The tool exports several functions that can be used by other tools:

```python
from clothing_recognition import (
    analyze_clothing,      # Core AI analysis function
    build_clothing_prompt, # Generate Gemini prompts
    format_for_audio,      # Format text for TTS
    resize_image_if_needed, # Image preprocessing
    convert_cv2_to_pil     # Format conversion
)
```

## Integration with ProgramAT App
When used within the ProgramAT mobile app, the tool automatically:
- Receives camera frames from the app
- Processes images on the server
- Returns audio-friendly descriptions
- Supports text-to-speech output

The app handles all network communication. The tool simply processes the image and returns results.

## Performance Notes
- Images are automatically resized to 1024x1024 max for efficiency
- Processing typically takes 2-5 seconds depending on network speed
- Uses Gemini Flash model for fast, accurate results
- No GPU required - runs on CPU

## Privacy & Security
- Images are sent to Google's Gemini API for processing
- Images are not stored after processing
- API key should be kept secure and not shared
- Consider privacy implications when analyzing personal clothing items

## Troubleshooting

### "Gemini API not available" error
Install the required package:
```bash
pip install google-generativeai
```

### "Gemini API key not configured" error
Set your API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Poor quality results
- Ensure the image has good lighting and the clothing item is clearly visible
- The tool automatically focuses on the most prominent item
- All detail levels still produce descriptions under 15 words

## Comparison with Other Tools

| Tool | Purpose | Best For |
|------|---------|----------|
| **clothing_recognition.py** | Single clothing item analysis | Quick identification of the most prominent clothing (under 15 words) |
| object_recognition.py | General objects | Everyday objects in COCO dataset |
| scene_description.py | General scenes | Overall scene understanding |
| live_ocr.py | Text extraction | Reading text from images |

## Future Enhancements
Potential improvements for future versions:
- Brand/logo recognition
- Size estimation
- Condition assessment (new, worn, damaged)
- Style matching and outfit coordination
- Shopping assistance features
- Fabric type identification

## Support
For issues or questions:
1. Check the error message for specific guidance
2. Verify your Gemini API key is set correctly
3. Ensure all dependencies are installed
4. Review the example usage above

## License
Part of the ProgramAT project. See main repository for license details.
