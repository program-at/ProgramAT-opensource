"""
Test script for clothing_recognition.py
Tests the clothing recognition tool with sample data
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from clothing_recognition import (
    analyze_clothing,
    build_clothing_prompt,
    format_for_audio,
    resize_image_if_needed,
    convert_cv2_to_pil,
    main
)

def create_test_image():
    """Create a simple test image with colored shapes simulating clothing"""
    # Create a blank image (white background)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a red rectangle (simulating a red shirt)
    cv2.rectangle(image, (100, 50), (300, 250), (0, 0, 255), -1)
    
    # Draw blue rectangle (simulating blue pants)
    cv2.rectangle(image, (100, 260), (300, 450), (255, 0, 0), -1)
    
    # Add some text to simulate a graphic print
    cv2.putText(image, "LOGO", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image

def test_resize_image():
    """Test image resizing"""
    print("Testing resize_image_if_needed()...")
    image = create_test_image()
    
    print(f"  Original size: {image.shape}")
    
    # Test with smaller max size
    resized = resize_image_if_needed(image, max_size=(320, 240))
    print(f"  Resized to: {resized.shape}")
    
    # Test with larger max size (should not resize)
    not_resized = resize_image_if_needed(image, max_size=(1920, 1080))
    print(f"  Max size larger: {not_resized.shape} (should be same as original)")
    print()

def test_build_prompt():
    """Test prompt building"""
    print("Testing build_clothing_prompt()...")
    
    for level in ['brief', 'standard', 'detailed']:
        prompt = build_clothing_prompt(level)
        print(f"  {level.capitalize()} prompt length: {len(prompt)} chars")
        print(f"  First 100 chars: {prompt[:100]}...")
    print()

def test_format_for_audio():
    """Test audio formatting"""
    print("Testing format_for_audio()...")
    
    # Test with normal text (under 15 words)
    text1 = "Red t-shirt with graphic print."
    formatted1 = format_for_audio(text1)
    word_count1 = len(formatted1.split())
    print(f"  Normal text: '{formatted1}' ({word_count1} words)")
    
    # Test with long text (should truncate to 15 words)
    text2 = "I see a very bright red t-shirt with a large graphic print showing a detailed cartoon character logo on the front center chest area with vibrant colors and intricate design elements."
    formatted2 = format_for_audio(text2, max_words=15)
    word_count2 = len(formatted2.split())
    print(f"  Long text (truncated): '{formatted2}' ({word_count2} words, max 15)")
    
    # Test with newlines
    text3 = "Red t-shirt\n\nwith\n\ngraphic print."
    formatted3 = format_for_audio(text3)
    word_count3 = len(formatted3.split())
    print(f"  With newlines: '{formatted3}' ({word_count3} words)")
    
    # Verify word count limit
    assert word_count2 <= 15, f"Word count should be <= 15, got {word_count2}"
    print(f"  ✓ Word count limit enforced")
    print()

def test_main_function():
    """Test main function with sample image"""
    print("Testing main() function...")
    image = create_test_image()
    
    # Test with default parameters
    print("  Testing with default parameters...")
    result = main(image)
    print(f"  Result type: {type(result)}")
    print(f"  Success: {result.get('success', False)}")
    print(f"  Description: {result.get('description', 'N/A')[:200]}")
    print(f"  Audio type: {result.get('audio', {}).get('type', 'N/A')}")
    print()
    
    # Test with brief detail level
    print("  Testing with brief detail level...")
    result_brief = main(image, {'detail_level': 'brief'})
    print(f"  Success: {result_brief.get('success', False)}")
    print(f"  Description: {result_brief.get('description', 'N/A')[:200]}")
    print()
    
    # Test with no image
    print("  Testing with no image...")
    result_no_image = main(None)
    print(f"  Success: {result_no_image.get('success', False)}")
    print(f"  Description: {result_no_image.get('description', 'N/A')}")
    print()

def test_analyze_clothing():
    """Test clothing analysis function"""
    print("Testing analyze_clothing()...")
    image = create_test_image()
    
    # Note: This test requires GEMINI_API_KEY to be set
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set, skipping live API test")
        print("  Set GEMINI_API_KEY environment variable to test with real API")
    else:
        print("  API key found, testing with Gemini...")
        result = analyze_clothing(image)
        print(f"  Success: {result['success']}")
        print(f"  Confidence: {result['confidence']}")
        if result['success']:
            print(f"  Description: {result['description'][:200]}...")
    print()

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CLOTHING RECOGNITION TOOL - TEST SUITE")
    print("=" * 60)
    print()
    
    test_resize_image()
    test_build_prompt()
    test_format_for_audio()
    test_main_function()
    test_analyze_clothing()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    run_all_tests()
