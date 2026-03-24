"""
Scene Description Tool for Blind Users

AI-powered scene description tool that generates detailed audio descriptions of images 
for blind or low vision users, similar to BeMyAI.

Features:
- Uses Google Gemini Vision API for intelligent scene analysis
- Audio-optimized output for text-to-speech
- Flexible detail levels (brief, standard, detailed)
- Multiple focus areas (general, people, objects, text, navigation)
- Building block architecture for extensibility
- Efficient image processing

Audio Output:
- Returns natural language descriptions suitable for text-to-speech
- Example: "A sunny park with people sitting on benches near a fountain."
"""

import cv2
import numpy as np
from typing import Dict, Optional, Any
import os
from PIL import Image
import re

# Constants
GEMINI_CONFIDENCE_SCORE = 0.9  # Gemini provides high-quality results

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Install with: pip install google-generativeai")


def get_scene_context(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract basic scene context without AI (lighting, colors, dimensions).
    
    Args:
        image: OpenCV image (numpy array in BGR format)
    
    Returns:
        Dictionary with scene context:
        {
            'width': int,
            'height': int,
            'brightness': str ('very dark', 'dark', 'normal', 'bright', 'very bright'),
            'dominant_color': str,
            'lighting': str ('very dark', 'dim', 'normal', 'bright', 'very bright')
        }
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale for brightness analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # Categorize brightness
    if avg_brightness < 50:
        brightness_level = 'very dark'
    elif avg_brightness < 100:
        brightness_level = 'dark'
    elif avg_brightness < 180:
        brightness_level = 'normal'
    elif avg_brightness < 230:
        brightness_level = 'bright'
    else:
        brightness_level = 'very bright'
    
    # Calculate dominant color (simple approach)
    if len(image.shape) == 3:
        avg_color = np.mean(image, axis=(0, 1))
        b, g, r = avg_color
        
        # Simple color classification
        if max(r, g, b) - min(r, g, b) < 30:
            if avg_brightness < 100:
                dominant_color = 'dark'
            elif avg_brightness > 180:
                dominant_color = 'bright'
            else:
                dominant_color = 'gray'
        elif r > g and r > b:
            dominant_color = 'reddish'
        elif g > r and g > b:
            dominant_color = 'greenish'
        elif b > r and b > g:
            dominant_color = 'bluish'
        else:
            dominant_color = 'mixed'
    else:
        dominant_color = 'grayscale'
    
    return {
        'width': width,
        'height': height,
        'brightness': brightness_level,
        'dominant_color': dominant_color,
        'lighting': brightness_level
    }


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


def build_scene_prompt(
    detail_level: str = 'standard',
    focus: str = 'general',
    context: Optional[Dict] = None
) -> str:
    """
    Build customized prompts for scene description based on parameters.
    
    Args:
        detail_level: 'brief', 'standard', or 'detailed'
        focus: 'general', 'people', 'objects', 'text', or 'navigation'
        context: Optional scene context from get_scene_context()
    
    Returns:
        Formatted prompt string for Gemini
    """
    # CRITICAL: Output will be read aloud via text-to-speech with a strict character limit.
    # Responses MUST be concise to fit within the audio output constraints.
    
    # Detail level instructions with strict length limits
    if detail_level == 'brief':
        length_limit = "STRICT LIMIT: Maximum 2 sentences, under 25 words total."
    elif detail_level == 'detailed':
        length_limit = "STRICT LIMIT: Maximum 4 sentences, under 50 words total."
    else:  # standard
        length_limit = "STRICT LIMIT: Maximum 3 sentences, under 40 words total."
    
    # Focus area instructions
    focus_instructions = {
        'people': "Focus on people - positions and actions. ",
        'objects': "Focus on key objects and their arrangement. ",
        'text': "Read any visible text. ",
        'navigation': "Describe layout for navigation - obstacles and paths. ",
        'general': ""
    }
    
    focus_instruction = focus_instructions.get(focus, "")
    
    # Context-aware additions
    context_note = ""
    if context:
        if context.get('lighting') in ['very dark', 'dark']:
            context_note = "Mention low lighting. "
    
    # Build prompt with length constraint at the END (most important position)
    prompt = f"Describe this image for a blind user. {focus_instruction}{context_note}Be direct and concise. No bullet points. {length_limit}"
    
    return prompt


def format_description_for_audio(
    description: str,
    style: str = 'narrative',
    max_length: Optional[int] = None
) -> str:
    """
    Format text for optimal audio presentation.
    
    Args:
        description: Raw description text
        style: 'narrative' (flowing) or 'concise' (direct)
        max_length: Optional maximum character length
    
    Returns:
        Formatted description suitable for TTS
    """
    # Clean up the text
    formatted = description.strip()
    
    # Remove excessive newlines
    formatted = ' '.join(formatted.split('\n'))
    
    # Remove excessive spaces
    formatted = ' '.join(formatted.split())
    
    # Apply style
    if style == 'concise':
        # Make more direct - remove filler words efficiently using regex
        filler_pattern = r'\b(basically|essentially|actually|literally)\b'
        formatted = re.sub(filler_pattern, '', formatted)
        # Clean up any double spaces created
        formatted = ' '.join(formatted.split())
    
    # Apply length limit if specified
    if max_length and len(formatted) > max_length:
        # Truncate at sentence boundary if possible
        truncated = formatted[:max_length]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > max_length * 0.7:  # At least 70% of desired length
            formatted = formatted[:last_sentence_end + 1]
        else:
            # Ensure we don't exceed max_length with ellipsis
            max_text_len = max_length - 3  # Reserve space for '...'
            formatted = formatted[:max_text_len].rstrip() + '...'
    
    return formatted


def analyze_scene(
    image: np.ndarray,
    api_key: Optional[str] = None,
    detail_level: str = 'standard',
    focus: str = 'general',
    model_name: str = 'gemini-3-flash-preview'
) -> Dict[str, Any]:
    """
    Core function that performs AI-powered scene analysis using Gemini.
    
    Args:
        image: OpenCV image (numpy array in BGR format)
        api_key: Gemini API key (uses env var if not provided)
        detail_level: 'brief', 'standard', or 'detailed'
        focus: 'general', 'people', 'objects', 'text', or 'navigation'
        model_name: Gemini model to use
    
    Returns:
        Dictionary with analysis results:
        {
            'success': bool,
            'description': str,
            'confidence': float (0.0-1.0),
            'detail_level': str,
            'focus': str,
            'context': dict
        }
    """
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'description': 'Gemini API not available. Please install google-generativeai package.',
            'confidence': 0.0,
            'detail_level': detail_level,
            'focus': focus
        }
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if not api_key:
        return {
            'success': False,
            'description': 'Gemini API key not configured. Please set GEMINI_API_KEY environment variable.',
            'confidence': 0.0,
            'detail_level': detail_level,
            'focus': focus
        }
    
    try:
        # Get scene context
        context = get_scene_context(image)
        
        # Resize image if needed for efficiency
        processed_image = resize_image_if_needed(image, max_size=(1024, 1024))
        
        # Convert to PIL format
        pil_image = convert_cv2_to_pil(processed_image)
        
        # Build prompt
        prompt = build_scene_prompt(detail_level, focus, context)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        print(f"🤖 Using Gemini model: {model_name}")
        print(f"📋 Detail level: {detail_level}, Focus: {focus}")
        
        # Create model and generate description
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, pil_image])
        
        # Extract description
        description = response.text.strip()
        
        print(f"\n📋 Scene Description:\n{description}\n")
        
        return {
            'success': True,
            'description': description,
            'confidence': GEMINI_CONFIDENCE_SCORE,
            'detail_level': detail_level,
            'focus': focus,
            'context': context
        }
        
    except Exception as e:
        print(f"❌ Scene analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'description': f'Scene analysis failed: {str(e)}',
            'confidence': 0.0,
            'detail_level': detail_level,
            'focus': focus
        }


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main entry point for the scene description tool.
    Compatible with ProgramAT tool execution framework.
    
    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional configuration dictionary:
            - 'detail_level': 'brief', 'standard', or 'detailed' (default: 'standard')
            - 'focus': 'general', 'people', 'objects', 'text', or 'navigation' (default: 'general')
            - 'style': 'narrative' or 'concise' (default: 'narrative')
            - 'api_key': Optional API key override
            - 'model': Optional Gemini model override
    
    Returns:
        For simple string return (audio-friendly):
            "A sunny park with people sitting on benches near a fountain."
        
        For advanced dictionary return (with audio config):
            {
                'success': bool,
                'description': str (the scene description),
                'audio': {
                    'type': 'speech',
                    'text': str (what to speak),
                    'rate': float,
                    'interrupt': bool
                },
                'text': str (display text),
                'confidence': float,
                'detail_level': str,
                'focus': str
            }
    """
    # Handle no image case
    if image is None:
        return {
            'success': False,
            'description': 'No image available',
            'audio': {
                'type': 'error',
                'text': 'No camera image available for scene description',
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
    focus = input_data.get('focus', 'general')
    style = input_data.get('style', 'narrative')
    api_key = input_data.get('api_key')
    model = input_data.get('model', 'gemini-3-flash-preview')
    
    # Analyze scene
    result = analyze_scene(
        image=image,
        api_key=api_key,
        detail_level=detail_level,
        focus=focus,
        model_name=model
    )
    
    # Format description for audio
    if result['success']:
        formatted_description = format_description_for_audio(
            result['description'],
            style=style,
            max_length=350  # Limited for VoiceOver announceForAccessibility
        )
        
        print(f"🔊 Audio output ({len(formatted_description)} chars): {formatted_description}")
        
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
            'detail_level': detail_level,
            'focus': focus
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
            'detail_level': detail_level,
            'focus': focus
        }


# Building block exports for use by other tools
__all__ = [
    'main',
    'analyze_scene',
    'get_scene_context',
    'build_scene_prompt',
    'format_description_for_audio',
    'resize_image_if_needed',
    'convert_cv2_to_pil'
]
