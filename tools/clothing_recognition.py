"""
Clothing Category and Feature Recognition Tool

AI-powered clothing recognition tool that identifies the most prominent clothing 
item and describes its visual features concisely for blind or low vision users.

Features:
- Focuses on the biggest/most prominent clothing item in the image
- Identifies clothing categories (shirts, pants, shoes, dresses, jackets, etc.)
- Describes colors, patterns, and styles concisely
- Limited to 15 words for quick, audio-friendly output
- Uses Google Gemini Vision API for accurate recognition

Audio Output:
- Returns natural language descriptions suitable for text-to-speech
- Example: "Red t-shirt with graphic print."
"""

import cv2
import numpy as np
from typing import Dict, Optional, Any
import os
from PIL import Image

# Gemini availability check
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Install with: pip install google-generativeai")


# Building block functions from scene_description.py
def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    """
    Resize image efficiently while maintaining aspect ratio.
    
    Args:
        image: OpenCV image (numpy array)
        max_size: Maximum dimensions (width, height)
    
    Returns:
        Resized image if needed, original otherwise
    """
    height, width = image.shape[:2]
    max_width, max_height = max_size
    
    # Check if resize needed
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR image to PIL RGB format.
    
    Args:
        image: OpenCV image (numpy array in BGR format)
    
    Returns:
        PIL Image in RGB format
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL
    pil_image = Image.fromarray(image_rgb)
    
    return pil_image


def build_clothing_prompt(detail_level: str = 'standard') -> str:
    """
    Build customized prompt for clothing recognition.
    
    Args:
        detail_level: 'brief', 'standard', or 'detailed'
    
    Returns:
        Formatted prompt string for Gemini
    """
    # Base instruction - focus on the most prominent item only
    base = "Analyze the clothing in this image for a blind or low vision user. "
    
    # Focus instruction
    focus_instruction = "Focus ONLY on the single biggest or most prominent clothing item visible in the image. Ignore all other clothing items. "
    
    # Detail level instructions
    if detail_level == 'brief':
        detail_instruction = "Provide a very brief description including category and primary color. "
    elif detail_level == 'detailed':
        detail_instruction = "Describe the item's category, colors, patterns, textures, style details, and any visible logos or text. "
    else:  # standard
        detail_instruction = "Describe the item's category, color, pattern (if any), and notable visual features. "
    
    # Output format instructions - enforce 15 word limit
    format_instruction = (
        "IMPORTANT: Keep your entire response under 15 words. Be extremely concise.\n\n"
        "Include in natural language:\n"
        "1. Category (e.g., shirt, t-shirt, pants, jeans, dress, jacket, shoes)\n"
        "2. Color(s) (e.g., 'red', 'navy blue', 'gray')\n"
        "3. Pattern if notable (e.g., striped, graphic print, plain)\n"
        "4. One key feature if space allows (e.g., short sleeves, v-neck, logo)\n\n"
        "Example outputs:\n"
        "- 'Red t-shirt with graphic print.'\n"
        "- 'Blue denim jeans, straight leg.'\n"
        "- 'Navy striped button-down shirt.'\n"
        "- 'Black leather jacket with zipper.'"
    )
    
    # Combine all parts
    prompt = base + focus_instruction + detail_instruction + format_instruction
    
    return prompt


def analyze_clothing(
    image: np.ndarray,
    api_key: Optional[str] = None,
    detail_level: str = 'standard',
    model_name: str = 'gemini-3-flash-preview'
) -> Dict[str, Any]:
    """
    Core function that performs AI-powered clothing analysis using Gemini.
    
    Args:
        image: OpenCV image (numpy array in BGR format)
        api_key: Gemini API key (uses env var if not provided)
        detail_level: 'brief', 'standard', or 'detailed'
        model_name: Gemini model to use
    
    Returns:
        Dictionary with analysis results:
        {
            'success': bool,
            'description': str,
            'confidence': float (0.0-1.0),
            'detail_level': str
        }
    """
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'description': 'Gemini API not available. Please install google-generativeai package.',
            'confidence': 0.0,
            'detail_level': detail_level
        }
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if not api_key:
        return {
            'success': False,
            'description': 'Gemini API key not configured. Please set GEMINI_API_KEY environment variable.',
            'confidence': 0.0,
            'detail_level': detail_level
        }
    
    try:
        # Resize image if needed for efficiency
        processed_image = resize_image_if_needed(image, max_size=(1024, 1024))
        
        # Convert to PIL format
        pil_image = convert_cv2_to_pil(processed_image)
        
        # Build prompt
        prompt = build_clothing_prompt(detail_level)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        print(f"🤖 Using Gemini model: {model_name}")
        print(f"📋 Detail level: {detail_level}")
        
        # Create model and generate description
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, pil_image])
        
        # Extract description
        description = response.text.strip()
        
        print(f"\n👔 Clothing Analysis:\n{description}\n")
        
        return {
            'success': True,
            'description': description,
            'confidence': 0.9,  # Gemini provides high-quality results
            'detail_level': detail_level
        }
        
    except Exception as e:
        print(f"❌ Clothing analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'description': f'Clothing analysis failed: {str(e)}',
            'confidence': 0.0,
            'detail_level': detail_level
        }


def format_for_audio(description: str, max_words: int = 15) -> str:
    """
    Format clothing description for optimal audio presentation.
    Limited to 15 words for concise audio output.
    
    Args:
        description: Raw description text
        max_words: Maximum word count (default: 15)
    
    Returns:
        Formatted description suitable for TTS, under 15 words
    """
    # Clean up the text
    formatted = description.strip()
    
    # Remove excessive newlines and spaces
    formatted = ' '.join(formatted.split('\n'))
    formatted = ' '.join(formatted.split())
    
    # Count words and truncate if needed
    words = formatted.split()
    
    if len(words) > max_words:
        # Truncate to max_words
        formatted = ' '.join(words[:max_words])
        
        # Remove trailing punctuation if incomplete
        if not formatted.endswith(('.', '!', '?')):
            formatted = formatted.rstrip('.,;:') + '.'
    
    return formatted


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main entry point for the clothing recognition tool.
    Compatible with ProgramAT tool execution framework.
    
    Focuses on the single biggest/most prominent clothing item in the image.
    Limits descriptions to 15 words for concise audio output.
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration dictionary:
            - 'detail_level': 'brief', 'standard', or 'detailed' (default: 'standard')
            - 'api_key': Optional API key override
            - 'model': Optional Gemini model override
    
    Returns:
        Dictionary with analysis results and audio configuration:
        {
            'success': bool,
            'description': str (the clothing description, under 15 words),
            'audio': {
                'type': 'speech' or 'error',
                'text': str (what to speak),
                'rate': float,
                'interrupt': bool
            },
            'text': str (display text),
            'confidence': float,
            'detail_level': str
        }
        
        Examples of description text (all under 15 words):
        - "Red t-shirt with graphic print."
        - "Blue jeans, straight leg cut."
        - "Navy striped button-down shirt, long sleeves."
    """
    # Handle no image case
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'success': False,
            'description': 'No image available',
            'audio': {
                'type': 'error',
                'text': 'No camera image available for clothing recognition',
                'rate': 1.0,
                'interrupt': False
            },
            'text': 'No image available',
            'confidence': 0.0
        }
    
    # Parse input parameters
    if input_data is None:
        input_data = {}
    
    detail_level = input_data.get('detail_level', 'standard')
    api_key = input_data.get('api_key')
    model = input_data.get('model', 'gemini-3-flash-preview')
    
    # Analyze clothing
    result = analyze_clothing(
        image=image,
        api_key=api_key,
        detail_level=detail_level,
        model_name=model
    )
    
    # Format and return description
    if result['success']:
        # Limit to 15 words as per requirement
        formatted_description = format_for_audio(result['description'], max_words=15)
        
        # Return with audio configuration
        return {
            'success': True,
            'description': formatted_description,
            'audio': {
                'type': 'speech',
                'text': formatted_description,
                'rate': 1.0,
                'interrupt': False
            },
            'text': formatted_description,
            'confidence': result['confidence'],
            'detail_level': detail_level
        }
    else:
        # Return error with audio
        return {
            'success': False,
            'description': result['description'],
            'audio': {
                'type': 'error',
                'text': result['description'],
                'rate': 1.0,
                'interrupt': False
            },
            'text': result['description'],
            'confidence': 0.0,
            'detail_level': detail_level
        }


# Building block exports for use by other tools
__all__ = [
    'main',
    'analyze_clothing',
    'build_clothing_prompt',
    'format_for_audio',
    'resize_image_if_needed',
    'convert_cv2_to_pil'
]
