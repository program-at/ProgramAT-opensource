# Live OCR Tool Usage Guide

## Overview

The Live OCR tool provides real-time text recognition for blind users, similar to SeeingAI's short text feature. It reads visible text aloud quickly without lengthy AI descriptions.

## Key Features

- **Real-time text reading**: Instantly reads text visible in the camera frame
- **Avoids repetition**: Tracks previously read text and only speaks new content
- **Camera movement detection**: Automatically detects when user moves camera to new text
- **Audio-optimized output**: Returns text in small, natural chunks for clear speech
- **Smart chunking**: Breaks long text into digestible audio segments
- **Client caching**: Caches Vision API client for faster performance
- **Context change detection**: Clears history when moving to completely different text (< 30% similarity)

## How It Works

1. User points camera at text (e.g., milk bottle label)
2. Tool detects and reads: "Milk, 2%, Kirkland"
3. User holds camera steady - tool stays silent (no repetition)
4. User turns bottle to nutrition facts
5. Tool detects new text and reads: "Servings 10, Calories 50"
6. User moves to newspaper
7. Tool reads headline: "New York Times, January 14"

## Configuration

### API Setup

The tool uses Google Cloud Vision API for high-quality OCR. Set up using one of these methods:

```bash
# Method 1 (Recommended): Use standard Google Cloud credentials file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Method 2: Use Google Cloud Vision API key
export GOOGLE_CLOUD_VISION_API_KEY="your-api-key-here"

# Method 3: Use generic Google API key
export GOOGLE_API_KEY="your-api-key-here"
```

The tool checks for credentials in this order:
1. `GOOGLE_APPLICATION_CREDENTIALS` (standard Google Cloud variable, points to JSON file)
2. `GOOGLE_CLOUD_VISION_API_KEY` (API key as string)
3. `GOOGLE_API_KEY` (fallback API key)

**Fallback**: If Google Cloud Vision is not available, the tool falls back to Tesseract OCR (if installed).

### Input Parameters

Configure the tool via `input_data` dictionary:

```python
input_data = {
    'chunk_size': 10,              # Max words per audio chunk (default: 10)
    'similarity_threshold': 0.8,    # Text similarity to avoid repetition (default: 0.8)
    'min_confidence': 0.5,          # Minimum OCR confidence (default: 0.5)
    'track_mode': True,             # Enable text tracking (default: True)
    'language': 'en',               # Language hint for OCR (default: 'en')
    'reset': False,                 # Reset text tracking (default: False)
    'api_key': None                 # Optional API key override
}
```

### Configuration Options Explained

- **chunk_size**: Controls how many words are spoken together. Smaller values = more pauses, larger values = longer sentences.
  
- **similarity_threshold**: How similar new text must be to previous text to be considered a duplicate (0.0 = completely different required, 1.0 = exact match required). Default 0.8 means 80% similar text is considered duplicate.

- **min_confidence**: Minimum OCR confidence to accept detected text. Higher values = fewer false positives but might miss some text.

- **track_mode**: When `True`, the tool remembers what it has already read and stays silent for repeated text. When `False`, it always reads all detected text.

- **language**: Language hint for the OCR engine. Use standard language codes: 'en', 'es', 'fr', 'de', etc.

- **reset**: Set to `True` to clear the text history and start fresh.

## Usage Examples

### Example 1: Basic Usage (from Mobile App)

The mobile app automatically streams camera frames to the tool:

```python
# Mobile app sends frame with default settings
# Tool receives frame and returns audio description
result = main(camera_frame, {})

# Output: "Milk, 2%, Kirkland"
```

### Example 2: Custom Chunk Size

For shorter audio segments:

```python
result = main(camera_frame, {
    'chunk_size': 5,  # Max 5 words per chunk
    'track_mode': True
})

# Long text will be split: "New York Times" then "January 14 2026"
```

### Example 3: Non-Tracking Mode

Always read text, even if repeated:

```python
result = main(camera_frame, {
    'track_mode': False
})

# Will read text every frame, even if it's the same text
```

### Example 4: Different Language

For non-English text:

```python
result = main(camera_frame, {
    'language': 'es',  # Spanish
    'track_mode': True
})
```

### Example 5: Resetting Tracking

Clear history when starting a new task:

```python
result = main(camera_frame, {
    'reset': True
})

# Output: "Text tracking reset"
# Next frame will read text as if it's the first time
```

## Return Format

### Simple Mode (track_mode=False, no new text)

Returns a string:
```
"Milk, 2%, Kirkland"
```

### Advanced Mode (with audio config)

Returns a dictionary:
```python
{
    'audio': {
        'type': 'speech',
        'text': 'Milk, 2%, Kirkland',
        'rate': 1.0,
        'interrupt': True  # New text interrupts current speech
    },
    'text': 'Milk, 2%, Kirkland',  # Display text
    'full_text': 'Milk 2% Kirkland'  # Raw OCR text
}
```

### Silent Mode (tracking enabled, same text)

Returns empty string:
```
""
```

This prevents repetitive audio playback during streaming.

## Building Block Functions

Other tools can use these exported functions:

### detect_text_google_vision(image, api_key, language_hints)
Performs OCR on an image using Google Cloud Vision API.

```python
from live_ocr import detect_text_google_vision

detections = detect_text_google_vision(
    image=camera_frame,
    api_key="your-key",
    language_hints=['en']
)
# Returns: [{'text': '...', 'confidence': 0.9, 'vertices': [...], 'locale': 'en'}]
```

### calculate_text_similarity(text1, text2)
Calculates similarity between two text strings (0.0 to 1.0).

```python
from live_ocr import calculate_text_similarity

similarity = calculate_text_similarity("Hello world", "Hello there")
# Returns: 0.5 (50% similar)
```

### chunk_text_for_audio(text, chunk_size)
Splits text into smaller chunks for better audio delivery.

```python
from live_ocr import chunk_text_for_audio

chunks = chunk_text_for_audio(
    "This is a very long piece of text that needs chunking",
    chunk_size=5
)
# Returns: ['This is a very long', 'piece of text that needs', 'chunking']
```

### format_text_for_speech(text)
Formats text for optimal TTS output.

```python
from live_ocr import format_text_for_speech

formatted = format_text_for_speech("Multiple    spaces   here")
# Returns: "Multiple spaces here"
```

## Performance Tips

1. **Chunk Size**: Use smaller chunks (5-7 words) for more natural pauses, larger chunks (10-15 words) for faster reading.

2. **Similarity Threshold**: 
   - Lower (0.6-0.7): More sensitive, reads text even if slightly changed
   - Higher (0.8-0.9): Less sensitive, only reads significantly different text

3. **Track Mode**: Always use `track_mode=True` for streaming to avoid repetition.

4. **API Choice**: Google Cloud Vision API provides better accuracy than Tesseract, especially for:
   - Small text
   - Low contrast text
   - Rotated text
   - Stylized fonts

## Real-Time Performance Optimizations

The tool includes optimizations to ensure real-time responsiveness:

### Context Change Priority
- Detects when text is completely different (< 30% similarity to history)
- Immediately clears history and speaks new text with interrupt=True
- Enables instant switching between different objects (milk → newspaper)
- Prioritizes new content when camera moves to different text

### Client Caching
- Vision API client is cached and reused across frames
- Eliminates expensive client creation overhead
- Reduces latency per OCR call

**Note:** The backend server already implements frame throttling (1 second default) to prevent overwhelming the OCR API. The tool works within this throttling to provide the best real-time experience possible.

## Performance Tips (continued)

5. **API Choice**: Google Cloud Vision API provides better accuracy than Tesseract, especially for:
   - Small text
   - Low contrast text
   - Rotated text
   - Stylized fonts

## Troubleshooting

### No text detected
- Ensure good lighting
- Hold camera steady
- Ensure text is in focus
- Check text size is readable (not too small)

### Text not updating
- Check similarity_threshold is not too low
- Ensure camera is actually moving to new text
- Try resetting tracking with `'reset': True`

### Repeated reading of same text
- Ensure `track_mode=True`
- Check similarity_threshold is not too high (try 0.7-0.8)

### API errors
- Verify API key is set correctly
- Check API quota/limits
- Ensure network connectivity

## Integration with Mobile App

The mobile app streams camera frames automatically. The tool:

1. Receives frame from camera
2. Detects text using OCR
3. Checks if text is new (not in history)
4. If new: returns audio description
5. If duplicate: returns empty string (silent)
6. Updates text history

This provides a smooth, real-time reading experience without repetition.
