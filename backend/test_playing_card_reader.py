"""
Test script for playing_card_reader.py
Tests the playing card reader tool with sample data
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from playing_card_reader import (
    identify_cards,
    build_card_prompt,
    format_for_audio,
    resize_image_if_needed,
    convert_cv2_to_pil,
    main,
)


def create_test_image():
    """Create a simple test image with colored rectangles simulating playing cards."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Draw two card-like rectangles side by side
    cv2.rectangle(image, (80, 100), (230, 380), (255, 255, 255), -1)
    cv2.rectangle(image, (80, 100), (230, 380), (0, 0, 0), 2)
    cv2.putText(image, "K", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "Hearts", (90, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.rectangle(image, (260, 100), (410, 380), (255, 255, 255), -1)
    cv2.rectangle(image, (260, 100), (410, 380), (0, 0, 0), 2)
    cv2.putText(image, "5", (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "Spades", (270, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return image


def test_resize_image():
    """Test image resizing."""
    print("Testing resize_image_if_needed()...")
    image = create_test_image()

    resized = resize_image_if_needed(image, max_size=(320, 240))
    print(f"  Resized to: {resized.shape}")

    not_resized = resize_image_if_needed(image, max_size=(1920, 1080))
    assert not_resized.shape == image.shape, "Image should not be resized when smaller than max_size"
    print(f"  No resize needed: {not_resized.shape} (matches original)")
    print()


def test_build_prompt():
    """Test prompt building."""
    print("Testing build_card_prompt()...")

    for streaming in [False, True]:
        prompt = build_card_prompt(streaming=streaming)
        label = "streaming" if streaming else "one-shot"
        print(f"  {label} prompt length: {len(prompt)} chars")
        assert 'playing card' in prompt.lower() or 'card' in prompt.lower()
        if streaming:
            assert '15 words' in prompt
        print(f"  First 100 chars: {prompt[:100]}...")
    print()


def test_format_for_audio():
    """Test audio formatting."""
    print("Testing format_for_audio()...")

    text1 = "King of hearts, 5 of spades."
    formatted1 = format_for_audio(text1)
    print(f"  Short text: '{formatted1}'")

    long_text = "Ace of spades, two of hearts, three of diamonds, four of clubs, five of spades, six of hearts."
    formatted2 = format_for_audio(long_text, max_words=15)
    word_count = len(formatted2.split())
    print(f"  Truncated (max 15): '{formatted2}' ({word_count} words)")
    assert word_count <= 15, f"Expected <= 15 words, got {word_count}"
    print(f"  ✓ Word count limit enforced")

    text3 = "King  of\n\nhearts."
    formatted3 = format_for_audio(text3)
    assert '\n' not in formatted3
    print(f"  Newlines removed: '{formatted3}'")
    print()


def test_main_no_image():
    """Test main() with no image."""
    print("Testing main() with no image...")
    result = main(None)
    assert isinstance(result, dict), "Should return dict on no image"
    assert result.get('audio', {}).get('type') == 'error'
    print(f"  ✓ Returns error dict: {result['audio']['text'][:60]}")
    print()


def test_main_with_image():
    """Test main() with a sample image (no API key needed for structure check)."""
    print("Testing main() with sample image (no API)...")
    image = create_test_image()

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set — testing error-handling path only")
        result = main(image)
        print(f"  Result type: {type(result).__name__}")
        if isinstance(result, dict):
            print(f"  Audio type: {result.get('audio', {}).get('type', 'N/A')}")
        else:
            print(f"  Result: {str(result)[:100]}")
    else:
        print("  API key found — running live Gemini test...")
        result = main(image)
        print(f"  Result type: {type(result).__name__}")
        if isinstance(result, str):
            print(f"  Output: {result}")
        elif isinstance(result, dict):
            print(f"  Audio: {result.get('audio', {}).get('text', 'N/A')[:100]}")
    print()


def test_main_streaming():
    """Test main() in streaming mode."""
    print("Testing main() in streaming mode...")
    image = create_test_image()
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set — skipping live streaming test")
        return
    result = main(image, {'streaming': True})
    if isinstance(result, str) and result:
        word_count = len(result.split())
        print(f"  Streaming output ({word_count} words): '{result}'")
        assert word_count <= 15, f"Streaming output should be <= 15 words, got {word_count}"
        print(f"  ✓ Word count within streaming limit")
    else:
        print(f"  Result: {result}")
    print()


def test_identify_cards():
    """Test identify_cards() with live API if available."""
    print("Testing identify_cards()...")
    image = create_test_image()
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set — skipping live API test")
        result = identify_cards(image, api_key='')
        assert not result['success']
        print(f"  ✓ Returns failure without API key: {result['description'][:60]}")
    else:
        result = identify_cards(image, api_key=api_key)
        print(f"  Success: {result['success']}")
        print(f"  Cards found: {result['cards_found']}")
        print(f"  Description: {result['description'][:200]}")
    print()


def run_all_tests():
    print("=" * 60)
    print("PLAYING CARD READER TOOL - TEST SUITE")
    print("=" * 60)
    print()

    test_resize_image()
    test_build_prompt()
    test_format_for_audio()
    test_main_no_image()
    test_main_with_image()
    test_main_streaming()
    test_identify_cards()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
