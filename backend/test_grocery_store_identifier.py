"""
Test script for grocery_store_identifier.py
Tests the grocery store item and price identifier tool with sample data.
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from grocery_store_identifier import (
    _calculate_similarity,
    _is_duplicate,
    _build_prompt,
    _resize_image,
    _convert_to_pil,
    reset_tracking,
    main,
)


def create_shelf_image():
    """Create a simple test image simulating a grocery shelf with text labels."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220  # light gray background

    # Draw shelf surface
    cv2.rectangle(image, (0, 300), (640, 320), (100, 80, 60), -1)

    # Draw three boxes on the shelf
    colors = [(0, 128, 255), (0, 200, 0), (200, 50, 50)]
    labels = ["Cheerios", "Lucky Charms", "Raisin Bran"]
    prices = ["$3.50", "$4.00", "$2.50"]
    x_starts = [20, 220, 420]

    for i in range(3):
        x = x_starts[i]
        cv2.rectangle(image, (x, 100), (x + 170, 298), colors[i], -1)
        cv2.putText(image, labels[i], (x + 5, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, prices[i], (x + 5, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    return image


def create_grabbed_item_image():
    """Create a test image simulating a held item close to the camera."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 30  # dark background

    # One box filling most of the frame
    cv2.rectangle(image, (30, 20), (610, 460), (0, 128, 255), -1)
    cv2.putText(image, "Cheerios", (80, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(image, "$3.50", (150, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 5, cv2.LINE_AA)

    return image


def test_resize_image():
    """Test image resizing helper."""
    print("Testing _resize_image()...")
    image = create_shelf_image()
    print(f"  Original: {image.shape}")

    resized = _resize_image(image, max_size=(320, 240))
    print(f"  Resized (max 320x240): {resized.shape}")
    assert resized.shape[0] <= 240 and resized.shape[1] <= 320, "Resize failed"

    unchanged = _resize_image(image, max_size=(1920, 1080))
    assert unchanged.shape == image.shape, "Should not resize when within limits"
    print("  PASSED\n")


def test_convert_to_pil():
    """Test BGR to PIL conversion."""
    print("Testing _convert_to_pil()...")
    image = create_shelf_image()
    pil_img = _convert_to_pil(image)
    assert pil_img.mode == 'RGB', "Expected RGB PIL image"
    assert pil_img.size == (image.shape[1], image.shape[0]), "Size mismatch"
    print("  PASSED\n")


def test_calculate_similarity():
    """Test Jaccard similarity calculation."""
    print("Testing _calculate_similarity()...")

    sim1 = _calculate_similarity("Cheerios 3.50", "Cheerios 3.50")
    assert sim1 == 1.0, f"Expected 1.0, got {sim1}"

    sim2 = _calculate_similarity("Cheerios 3.50", "Lucky Charms 4 dollars")
    assert sim2 < 0.3, f"Expected low similarity, got {sim2}"

    sim3 = _calculate_similarity("", "anything")
    assert sim3 == 0.0, f"Expected 0.0 for empty string, got {sim3}"

    print(f"  Identical: {sim1:.2f}, Different: {sim2:.2f}, Empty: {sim3:.2f}")
    print("  PASSED\n")


def test_duplicate_detection():
    """Test duplicate suppression logic."""
    print("Testing _is_duplicate()...")
    reset_tracking()

    from grocery_store_identifier import _result_history
    _result_history.append("Cheerios, 3.50")

    assert _is_duplicate("Cheerios, 3.50"), "Should detect exact duplicate"
    # Near-duplicate: same product with slightly different phrasing (enough shared words)
    assert _is_duplicate("Cheerios, 3.50 dollars"), "Should detect near-duplicate with extra word"
    assert not _is_duplicate("Lucky Charms, 4 dollars"), "Should not flag new item"
    reset_tracking()
    print("  PASSED\n")


def test_build_prompt():
    """Test that the prompt is well-formed."""
    print("Testing _build_prompt()...")
    prompt = _build_prompt()
    assert isinstance(prompt, str) and len(prompt) > 50, "Prompt too short"
    assert "blind" in prompt.lower(), "Prompt should mention blind user context"
    assert "grabbed" in prompt.lower(), "Prompt should handle grab scenario"
    assert "15 words" in prompt.lower(), "Prompt should enforce word limit"
    print(f"  Prompt length: {len(prompt)} chars")
    print("  PASSED\n")


def test_main_no_image():
    """Test main() with invalid image inputs."""
    print("Testing main() with invalid inputs...")
    reset_tracking()

    result = main(None)
    assert result == "No camera image available", f"Unexpected: {result}"

    result = main(np.array([]))
    assert result == "No camera image available", f"Unexpected: {result}"
    print("  PASSED\n")


def test_main_reset():
    """Test reset functionality."""
    print("Testing main() reset...")
    reset_tracking()

    image = create_shelf_image()
    result = main(image, {'reset': True})
    assert "reset" in result.lower(), f"Unexpected reset result: {result}"
    print(f"  Reset result: '{result}'")
    print("  PASSED\n")


def test_main_frame_skip():
    """Test that main() skips frames as configured."""
    print("Testing frame skip in main()...")
    reset_tracking()

    image = create_shelf_image()
    # With skip_frames=3, frames 1 and 2 should return ""
    result1 = main(image, {'skip_frames': 3})
    result2 = main(image, {'skip_frames': 3})
    assert result1 == "", f"Frame 1 should be skipped, got: '{result1}'"
    assert result2 == "", f"Frame 2 should be skipped, got: '{result2}'"
    print(f"  Frame 1: '{result1}' (expected '')")
    print(f"  Frame 2: '{result2}' (expected '')")
    print("  PASSED\n")


def test_main_with_shelf_image():
    """Test main() with a shelf image (requires Gemini API key)."""
    print("Testing main() with shelf image (live API call)...")
    reset_tracking()

    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print("  SKIPPED: GEMINI_API_KEY not set\n")
        return

    image = create_shelf_image()
    result = main(image, {'skip_frames': 1, 'api_key': api_key})
    print(f"  Result: '{result}'")

    if result:
        if isinstance(result, str):
            result_text = result
        elif isinstance(result, dict):
            result_text = result.get('text', '')
        else:
            result_text = str(result)
        word_count = len(result_text.split())
        print(f"  Word count: {word_count}")
        assert word_count <= 20, f"Response too long ({word_count} words): {result_text}"
    print("  PASSED\n")


def test_main_with_grabbed_image():
    """Test main() with a grabbed item image (requires Gemini API key)."""
    print("Testing main() with grabbed item image (live API call)...")
    reset_tracking()

    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print("  SKIPPED: GEMINI_API_KEY not set\n")
        return

    image = create_grabbed_item_image()
    result = main(image, {'skip_frames': 1, 'api_key': api_key})
    print(f"  Result: '{result}'")

    if result:
        if isinstance(result, dict):
            assert isinstance(result.get('audio'), dict), \
                "Grab result dict should contain 'audio' key"
            assert result['audio'].get('type') == 'success', \
                "Grab should use success audio type"
            print(f"  Audio type: {result['audio']['type']} (expected 'success')")
        elif isinstance(result, str):
            print(f"  String result (no grab detected): '{result}'")
    print("  PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Grocery Store Identifier — Test Suite")
    print("=" * 60 + "\n")

    test_resize_image()
    test_convert_to_pil()
    test_calculate_similarity()
    test_duplicate_detection()
    test_build_prompt()
    test_main_no_image()
    test_main_reset()
    test_main_frame_skip()
    test_main_with_shelf_image()
    test_main_with_grabbed_image()

    print("All tests completed.")
