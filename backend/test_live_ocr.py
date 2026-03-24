"""
Test script for live_ocr.py
Tests the live OCR tool with sample data
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from live_ocr import (
    detect_text_google_vision,
    calculate_text_similarity,
    chunk_text_for_audio,
    format_text_for_speech,
    is_text_similar_to_history,
    update_text_history,
    reset_text_tracking,
    main
)


def create_test_image_with_text(text: str, width: int = 640, height: int = 480):
    """Create a test image with text rendered on it"""
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Convert to PIL for text rendering
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a truetype font from common locations, fall back to default
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]
    
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 48)
            break
        except:
            continue
    
    if font is None:
        # Use default font if truetype not available
        font = ImageFont.load_default()
    
    # Calculate text position (centered)
    # For default font, we'll estimate positioning
    text_x = 50
    text_y = height // 2 - 50
    
    # Draw text in black
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    # Convert back to OpenCV format
    image = np.array(pil_image)
    
    return image


def test_text_similarity():
    """Test text similarity calculation"""
    print("Testing calculate_text_similarity()...")
    
    test_cases = [
        ("Hello world", "Hello world", 1.0),
        ("Hello world", "Hello there", 0.5),
        ("The quick brown fox", "The lazy brown dog", 0.4),
        ("", "Hello", 0.0),
        ("Completely different text", "Hello world", 0.0),
    ]
    
    for text1, text2, expected_range in test_cases:
        similarity = calculate_text_similarity(text1, text2)
        print(f"  '{text1}' vs '{text2}' → {similarity:.2f}")
        if abs(similarity - expected_range) < 0.2:
            print(f"    ✓ Close to expected {expected_range:.2f}")
        else:
            print(f"    ⚠ Expected ~{expected_range:.2f}")
    print()


def test_chunk_text():
    """Test text chunking for audio"""
    print("Testing chunk_text_for_audio()...")
    
    test_cases = [
        ("Hello world", 5, 1),  # Short text, 1 chunk
        ("This is a longer piece of text that should be split into multiple chunks", 5, 3),
        ("Word", 10, 1),
    ]
    
    for text, chunk_size, expected_chunks in test_cases:
        chunks = chunk_text_for_audio(text, chunk_size)
        print(f"  Text: '{text[:40]}...' (chunk_size={chunk_size})")
        print(f"    → {len(chunks)} chunks (expected ~{expected_chunks})")
        for i, chunk in enumerate(chunks):
            print(f"      Chunk {i+1}: '{chunk}'")
    print()


def test_format_text():
    """Test text formatting for speech"""
    print("Testing format_text_for_speech()...")
    
    test_cases = [
        "Hello   world",
        "Multiple    spaces    here",
        "A B C D E F",
        "Normal text with proper spacing",
    ]
    
    for text in test_cases:
        formatted = format_text_for_speech(text)
        print(f"  '{text}' → '{formatted}'")
    print()


def test_text_tracking():
    """Test text tracking and history"""
    print("Testing text tracking...")
    
    # Reset first
    reset_text_tracking()
    
    # Add some text to history
    texts = [
        "Hello world",
        "Good morning",
        "Hello there"
    ]
    
    for text in texts:
        update_text_history(text)
        print(f"  Added to history: '{text}'")
    
    # Test similarity checks
    print("\n  Checking similarity to history:")
    test_texts = [
        ("Hello world", True),  # Exact match
        ("Hello there world", True),  # Similar
        ("Goodbye world", False),  # Different
    ]
    
    for text, expected in test_texts:
        is_similar = is_text_similar_to_history(text, threshold=0.8)
        status = "✓" if is_similar == expected else "✗"
        print(f"    {status} '{text}' similar={is_similar} (expected={expected})")
    
    print()


def test_main_function():
    """Test main() function with sample images"""
    print("Testing main() function...")
    
    # Test 1: Image with text
    print("\n  Test 1: Image with clear text")
    image1 = create_test_image_with_text("MILK 2% KIRKLAND")
    result1 = main(image1, {'track_mode': False})
    print(f"    Result: {result1}")
    
    # Test 2: Same text again with tracking (should return empty)
    print("\n  Test 2: Same text with tracking enabled")
    reset_text_tracking()
    result2a = main(image1, {'track_mode': True})
    print(f"    First call: {result2a}")
    result2b = main(image1, {'track_mode': True})
    print(f"    Second call (should be empty): '{result2b}'")
    
    # Test 3: Different text
    print("\n  Test 3: Different text with tracking")
    image3 = create_test_image_with_text("NEW YORK TIMES")
    result3 = main(image3, {'track_mode': True})
    print(f"    Result: {result3}")
    
    # Test 4: None image
    print("\n  Test 4: None image")
    result4 = main(None, {})
    print(f"    Result: {result4}")
    
    # Test 5: Empty image
    print("\n  Test 5: Empty image")
    empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result5 = main(empty_image, {'track_mode': False})
    print(f"    Result: {result5}")
    
    print()


def test_integration():
    """Integration test simulating real usage scenario"""
    print("Integration Test: Simulating milk bottle scanning scenario")
    print("-" * 60)
    
    # Reset tracking
    reset_text_tracking()
    
    # Scenario: User points at milk bottle
    print("\n1. User points at milk bottle label")
    milk_image = create_test_image_with_text("MILK 2% KIRKLAND")
    result1 = main(milk_image, {'track_mode': True, 'chunk_size': 5})
    print(f"   Audio output: {result1}")
    
    # Still looking at same label
    print("\n2. Still looking at milk bottle (should be silent)")
    result2 = main(milk_image, {'track_mode': True, 'chunk_size': 5})
    print(f"   Audio output: '{result2}' (should be empty)")
    
    # Turn bottle to nutrition facts
    print("\n3. Turn bottle to nutrition facts")
    nutrition_image = create_test_image_with_text("SERVINGS 10 CALORIES 50")
    result3 = main(nutrition_image, {'track_mode': True, 'chunk_size': 5})
    print(f"   Audio output: {result3}")
    
    # Move to newspaper
    print("\n4. Move away to newspaper")
    newspaper_image = create_test_image_with_text("NEW YORK TIMES JANUARY 14")
    result4 = main(newspaper_image, {'track_mode': True, 'chunk_size': 5})
    print(f"   Audio output: {result4}")
    
    # Look at newspaper again (should be silent)
    print("\n5. Still looking at newspaper (should be silent)")
    result5 = main(newspaper_image, {'track_mode': True, 'chunk_size': 5})
    print(f"   Audio output: '{result5}' (should be empty)")
    
    print("\n" + "-" * 60)
    print()


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("LIVE OCR TOOL TEST SUITE")
    print("=" * 60)
    print()
    
    test_text_similarity()
    test_chunk_text()
    test_format_text()
    test_text_tracking()
    test_main_function()
    test_integration()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print()
    print("Note: Actual OCR results depend on Google Cloud Vision API")
    print("or Tesseract availability. Some tests may show limited")
    print("results if neither is configured.")


if __name__ == "__main__":
    run_all_tests()
