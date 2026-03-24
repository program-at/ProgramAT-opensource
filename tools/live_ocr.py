"""
Live OCR Tool for Blind Users

Real-time text recognition tool that reads aloud visible text for blind users.
Similar to SeeingAI short text feature - provides quick text reading without
lengthy AI descriptions.

Features:
- Real-time OCR using Google Cloud Vision API
- Avoids repetition by tracking previously read text
- Detects camera movement and reads new text
- Returns text in small, speakable chunks
- Audio-optimized output for text-to-speech

Usage from Mobile App:
The tool automatically receives camera frames from the mobile app and returns
audio feedback with detected text.

Example Return Format:
Simple mode:
"Milk, 2%, Kirkland"

Advanced mode (with dict):
{
    'audio': {
        'type': 'speech',
        'text': 'Milk, 2%, Kirkland',
        'rate': 1.0,
        'interrupt': True
    },
    'text': 'Milk, 2%, Kirkland',
    'full_text': 'Full detected text...'
}

Configuration Options (via input_data):
- chunk_size: Maximum words per audio chunk (default 6)
- similarity_threshold: Text similarity threshold to avoid repetition (default 0.8)
- min_confidence: Minimum OCR confidence (default 0.5)
- track_mode: Enable text tracking across frames (default True)
- language: Language hint for OCR (default 'en')

Building Block Functions:
This tool exports several functions that can be used by other tools:
- detect_text_google_vision(image, api_key, language_hints) -> List[Dict]
- calculate_text_similarity(text1, text2) -> float
- chunk_text_for_audio(text, chunk_size) -> List[str]
- format_text_for_speech(text) -> str
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
import os
import re
import base64
from collections import deque

# Default configuration
DEFAULT_CHUNK_SIZE = 6  # words per chunk (balanced for audio delivery and context switching)
DEFAULT_SIMILARITY_THRESHOLD = 0.8  # 80% similarity to consider text as duplicate
DEFAULT_MIN_CONFIDENCE = 0.5
MAX_HISTORY_SIZE = 5  # Keep last 5 text readings in memory

# Global state for text tracking across frames
_text_history = deque(maxlen=MAX_HISTORY_SIZE)
_last_spoken_text = ""
_frame_counter = 0
_ocr_unavailable_notified = False  # Track if we've already notified about OCR unavailability
_vision_client = None  # Cache the Vision API client to avoid recreating it
_vision_client_key = None  # Track which API key was used for the client

# Try to import Google Cloud Vision
try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False


def detect_text_google_vision(
    image: np.ndarray,
    api_key: Optional[str] = None,
    language_hints: List[str] = ['en']
) -> List[Dict[str, Any]]:
    """
    Detect text in an image using Google Cloud Vision API.
    
    Args:
        image: OpenCV image (numpy array in BGR format)
        api_key: Path to credentials JSON file, credentials as JSON string, or API key
                 If not provided, checks environment variables in order:
                 1. GOOGLE_APPLICATION_CREDENTIALS (path to JSON file)
                 2. GOOGLE_CLOUD_VISION_API_KEY (API key string)
                 3. GOOGLE_API_KEY (fallback API key)
        language_hints: List of language codes to hint the OCR engine
        
    Returns:
        List of detected text blocks, each containing:
            - text: The detected text string
            - confidence: Detection confidence (0.0 to 1.0)
            - vertices: Bounding box coordinates
            - locale: Detected language locale
            
        Returns special error dict if OCR is unavailable:
            - error: Error message
    """
    if not VISION_API_AVAILABLE:
        # Fallback to Tesseract OCR if available
        return _detect_text_tesseract(image)
    
    # Get API key or credentials file path
    if api_key is None:
        # First, check for GOOGLE_APPLICATION_CREDENTIALS (standard Google Cloud variable)
        api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if not api_key:
            # Fall back to GOOGLE_CLOUD_VISION_API_KEY
            api_key = os.environ.get('GOOGLE_CLOUD_VISION_API_KEY', '')
        if not api_key:
            # Fall back to GOOGLE_API_KEY
            api_key = os.environ.get('GOOGLE_API_KEY', '')
    
    if not api_key:
        # No API key configured - try Tesseract fallback
        return _detect_text_tesseract(image)
    
    try:
        # Cache Vision API client to avoid recreating it on every call (expensive operation)
        # Google Cloud Vision supports multiple authentication methods:
        # 1. GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON file
        # 2. JSON credentials passed directly as a string
        # 3. API key (less common for Vision API)
        from google.oauth2 import service_account
        import json
        from pathlib import Path
        
        global _vision_client, _vision_client_key
        
        client = None
        
        # Check if we can reuse cached client
        if _vision_client and _vision_client_key == api_key:
            client = _vision_client
        else:
            # Need to create new client
            # Check if api_key is a file path (for GOOGLE_APPLICATION_CREDENTIALS)
            if os.path.isfile(api_key):
                # It's a file path - load credentials from file
                credentials = service_account.Credentials.from_service_account_file(api_key)
                client = vision.ImageAnnotatorClient(credentials=credentials)
            elif api_key.startswith('{'):
                # It's JSON credentials as a string
                credentials_dict = json.loads(api_key)
                credentials = service_account.Credentials.from_service_account_info(credentials_dict)
                client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                # It's an API key or unknown format - try default client
                # The default client will use GOOGLE_APPLICATION_CREDENTIALS env var if set
                client = vision.ImageAnnotatorClient()
            
            # Cache the client
            _vision_client = client
            _vision_client_key = api_key
        
        # Convert image to bytes
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            return []
        
        content = encoded_image.tobytes()
        
        # Create Vision API image object
        vision_image = vision.Image(content=content)
        
        # Set up image context with language hints
        image_context = vision.ImageContext(language_hints=language_hints)
        
        # Perform text detection
        response = client.text_detection(image=vision_image, image_context=image_context)
        
        if response.error.message:
            raise Exception(response.error.message)
        
        texts = response.text_annotations
        
        if not texts:
            return []
        
        # First annotation contains full text, rest are individual words/blocks
        # We'll use the full text for simplicity and natural reading
        detections = []
        
        for text in texts:
            # Get confidence from text annotation if available
            # Note: Google Cloud Vision text_detection doesn't provide per-annotation confidence
            # For more granular confidence, use document_text_detection instead
            # For this use case, we assume high confidence for returned results
            # and let min_confidence filtering happen at a higher level
            confidence = 0.9  # High confidence assumption for returned results
            
            # Get bounding box vertices
            vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            
            # Get locale if available
            locale = text.locale if text.locale else 'en'
            
            detections.append({
                'text': text.description,
                'confidence': confidence,
                'vertices': vertices,
                'locale': locale
            })
        
        return detections
        
    except Exception as e:
        print(f"❌ Google Cloud Vision API error: {e}")
        # Fallback to Tesseract
        return _detect_text_tesseract(image)


def _detect_text_tesseract(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Fallback text detection using Tesseract OCR.
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        List of detected text blocks, or special error dict if unavailable
    """
    try:
        import pytesseract
        
        # Convert to RGB for Tesseract
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Get detailed text data
        data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)
        
        detections = []
        
        # Combine text by lines
        current_text = []
        current_conf = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
            
            if text:
                current_text.append(text)
                current_conf.append(conf / 100.0)  # Normalize to 0-1
        
        if current_text:
            full_text = ' '.join(current_text)
            avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
            
            detections.append({
                'text': full_text,
                'confidence': avg_conf,
                'vertices': [],
                'locale': 'en'
            })
        
        return detections
        
    except ImportError:
        # Return special error indicator - OCR is completely unavailable
        return [{'error': 'OCR not available. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable or install Tesseract.'}]
    except Exception as e:
        print(f"⚠️  Tesseract OCR error: {e}")
        return []


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using simple word overlap.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score (0.0 to 1.0), where 1.0 means identical
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    similarity = len(intersection) / len(union) if union else 0.0
    
    return similarity


def chunk_text_for_audio(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Split text into smaller chunks for better audio delivery.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum number of words per chunk
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Clean up text
    text = text.strip()
    
    # Split into words
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    # Split into chunks
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def format_text_for_speech(text: str) -> str:
    """
    Format text for optimal speech synthesis.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Formatted text suitable for TTS
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Add natural pauses for commas (already present in text)
    # No modification needed for commas
    
    # Clean up common OCR artifacts
    # Remove single character artifacts that are likely noise
    words = text.split()
    cleaned_words = []
    for word in words:
        # Keep single letters if they're likely intentional (A, I, etc.)
        if len(word) == 1 and word.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            cleaned_words.append(word)
        elif len(word) > 1:
            cleaned_words.append(word)
    
    text = ' '.join(cleaned_words)
    
    return text


def is_text_similar_to_history(text: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    """
    Check if text is similar to recently spoken text.
    
    Args:
        text: Text to check
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        True if text is similar to recent history, False otherwise
    """
    global _text_history
    
    for historical_text in _text_history:
        similarity = calculate_text_similarity(text, historical_text)
        if similarity >= threshold:
            return True
    
    return False


def is_text_completely_new(text: str, low_similarity_threshold: float = 0.3) -> bool:
    """
    Check if text is completely new (very low similarity to all history).
    This is used to detect context changes (e.g., moving from milk bottle to newspaper).
    
    Args:
        text: Text to check
        low_similarity_threshold: Threshold below which text is considered completely new
        
    Returns:
        True if text is completely different from all history, False otherwise
    """
    global _text_history
    
    if not _text_history:
        return True
    
    # Check similarity against all history items
    for historical_text in _text_history:
        similarity = calculate_text_similarity(text, historical_text)
        if similarity >= low_similarity_threshold:
            # Found at least one item with some similarity
            return False
    
    # No items in history have significant similarity - this is completely new
    return True


def update_text_history(text: str):
    """
    Update the text history with newly spoken text.
    
    Args:
        text: Text that was just spoken
    """
    global _text_history, _last_spoken_text
    
    _text_history.append(text)
    _last_spoken_text = text


def reset_text_tracking():
    """Reset text tracking state."""
    global _text_history, _last_spoken_text, _frame_counter, _ocr_unavailable_notified
    
    _text_history.clear()
    _last_spoken_text = ""
    _frame_counter = 0
    _ocr_unavailable_notified = False


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Main entry point for live OCR tool.
    
    Detects and reads text in camera frames, avoiding repetition and
    providing audio-friendly output in small chunks.
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration:
            - chunk_size: Max words per chunk (default 6)
            - similarity_threshold: Threshold to avoid repetition (default 0.8)
            - min_confidence: Minimum OCR confidence (default 0.5)
            - track_mode: Enable text tracking (default True)
            - language: Language hint for OCR (default 'en')
            - reset: Reset text tracking (default False)
            - api_key: Optional Google Cloud Vision API key override
    
    Returns:
        Audio-friendly text string or dictionary with audio configuration.
        Returns empty string "" when no new text is detected to avoid
        repetitive audio in streaming mode.
    """
    global _frame_counter, _ocr_unavailable_notified
    _frame_counter += 1
    
    # Handle None or invalid image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"
    
    # Parse input_data
    config = {}
    if isinstance(input_data, dict):
        config = input_data
    
    # Get configuration parameters
    chunk_size = config.get('chunk_size', DEFAULT_CHUNK_SIZE)
    similarity_threshold = config.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)
    min_confidence = config.get('min_confidence', DEFAULT_MIN_CONFIDENCE)
    track_mode = config.get('track_mode', True)
    language = config.get('language', 'en')
    api_key = config.get('api_key')
    
    # Handle reset request
    if config.get('reset', False):
        reset_text_tracking()
        _ocr_unavailable_notified = False
        return "Text tracking reset"
    
    # Detect text
    detections = detect_text_google_vision(
        image,
        api_key=api_key,
        language_hints=[language]
    )
    
    # Check if OCR is unavailable (special error case)
    if detections and len(detections) == 1 and 'error' in detections[0]:
        # OCR system is not available
        error_msg = detections[0]['error']
        
        # Only notify once to avoid spam
        if not _ocr_unavailable_notified:
            _ocr_unavailable_notified = True
            return {
                'audio': {
                    'type': 'error',
                    'text': error_msg,
                    'rate': 1.0,
                    'interrupt': True
                },
                'text': error_msg
            }
        else:
            # Already notified, return empty to avoid spam
            return ""
    
    # Filter by confidence
    detections = [d for d in detections if d.get('confidence', 0) >= min_confidence]
    
    if not detections:
        # No text detected (but OCR is working)
        if not track_mode:
            return "No text detected"
        else:
            return ""  # Silent in tracking mode
    
    # Get the full text (first detection from Vision API contains all text)
    full_text = detections[0]['text']
    
    if not full_text:
        if not track_mode:
            return "No text detected"
        else:
            return ""
    
    # Format text for speech
    formatted_text = format_text_for_speech(full_text)
    
    if not formatted_text:
        if not track_mode:
            return "No readable text detected"
        else:
            return ""
    
    # Handle tracking mode
    if track_mode:
        # First, check if this is completely new text (very different from history)
        # This handles the case where user moves camera to entirely different content
        completely_new = is_text_completely_new(formatted_text, low_similarity_threshold=0.3)
        
        if completely_new:
            # This is completely new context (e.g., moved from milk to newspaper)
            # Clear history to prioritize this new text and avoid interference from old text
            _text_history.clear()
            update_text_history(formatted_text)
            
            # Chunk text for better audio delivery
            chunks = chunk_text_for_audio(formatted_text, chunk_size)
            text_to_speak = ', '.join(chunks)
            
            # Return with interrupt=True to immediately read new text
            return {
                'audio': {
                    'type': 'speech',
                    'text': text_to_speak,
                    'rate': 1.0,
                    'interrupt': True  # Interrupt any ongoing speech for new context
                },
                'text': text_to_speak,
                'full_text': full_text
            }
        
        # Check if this text is similar to what we've already read
        if is_text_similar_to_history(formatted_text, similarity_threshold):
            # Same text, don't repeat
            return ""
        
        # New text detected (but within same context)!
        # Update history
        update_text_history(formatted_text)
        
        # Chunk text for better audio delivery
        chunks = chunk_text_for_audio(formatted_text, chunk_size)
        
        # Join chunks with commas for natural pauses in speech
        text_to_speak = ', '.join(chunks)
        
        # Return with advanced audio configuration for immediate reading
        return {
            'audio': {
                'type': 'speech',
                'text': text_to_speak,
                'rate': 1.0,
                'interrupt': True  # Interrupt previous speech for new text
            },
            'text': text_to_speak,
            'full_text': full_text
        }
    else:
        # Normal mode: always return detected text
        reset_text_tracking()
        
        # Chunk text
        chunks = chunk_text_for_audio(formatted_text, chunk_size)
        text_to_speak = ', '.join(chunks)
        
        return {
            'audio': {
                'type': 'speech',
                'text': text_to_speak,
                'rate': 1.0,
                'interrupt': False
            },
            'text': text_to_speak,
            'full_text': full_text
        }


# Building block exports for use by other tools
__all__ = [
    'main',
    'detect_text_google_vision',
    'calculate_text_similarity',
    'chunk_text_for_audio',
    'format_text_for_speech',
    'is_text_similar_to_history',
    'is_text_completely_new',
    'update_text_history',
    'reset_text_tracking'
]
