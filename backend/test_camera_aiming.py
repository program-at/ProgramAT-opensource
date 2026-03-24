"""
Test script for camera_aiming.py
Tests the camera aiming assistance tool with various scenarios
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from camera_aiming import (
    calculate_framing_metrics,
    generate_directional_cues,
    is_well_framed,
    select_target_object,
    reset_lock,
    main
)


def create_test_image_with_object(
    obj_center_x: float = 0.5,
    obj_center_y: float = 0.5,
    obj_size: float = 0.3,
    width: int = 640,
    height: int = 480
):
    """
    Create a test image with a colored rectangle at specified position.
    
    Args:
        obj_center_x: Normalized X position (0.0 = left, 1.0 = right)
        obj_center_y: Normalized Y position (0.0 = top, 1.0 = bottom)
        obj_size: Size as ratio of frame (e.g., 0.3 = 30% of frame)
        width: Image width
        height: Image height
    """
    # Create a blank image (white background)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Calculate object dimensions
    obj_width = int(width * obj_size)
    obj_height = int(height * obj_size)
    
    # Calculate object position
    center_x = int(width * obj_center_x)
    center_y = int(height * obj_center_y)
    
    x1 = center_x - obj_width // 2
    y1 = center_y - obj_height // 2
    x2 = x1 + obj_width
    y2 = y1 + obj_height
    
    # Draw a red rectangle (simulating an object)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), -1)
    
    return image


def test_framing_metrics():
    """Test framing metrics calculation"""
    print("Testing calculate_framing_metrics()...")
    
    # Test centered object
    bbox = [220, 140, 200, 200]  # Centered in 640x480
    metrics = calculate_framing_metrics(bbox, 640, 480)
    
    print(f"  Centered object metrics:")
    print(f"    Center: ({metrics['center_x']:.2f}, {metrics['center_y']:.2f})")
    print(f"    Size ratio: {metrics['size_ratio']:.2f}")
    print(f"    Horizontal offset: {metrics['horizontal_offset']:.2f}")
    print(f"    Vertical offset: {metrics['vertical_offset']:.2f}")
    print(f"    Is centered: {metrics['is_centered']}")
    print(f"    Is good size: {metrics['is_good_size']}")
    
    # Test off-center object (left)
    bbox = [50, 140, 200, 200]  # Left side
    metrics = calculate_framing_metrics(bbox, 640, 480)
    print(f"\n  Left-side object metrics:")
    print(f"    Horizontal offset: {metrics['horizontal_offset']:.2f}")
    print(f"    Is centered: {metrics['is_centered']}")
    print()


def test_directional_cues():
    """Test directional cue generation"""
    print("Testing generate_directional_cues()...")
    
    test_cases = [
        # (description, bbox, expected_keywords)
        ("Centered, good size", [220, 140, 200, 200], ["Hold", "steady"]),
        ("Too far left", [50, 140, 200, 200], ["right"]),
        ("Too far right", [390, 140, 200, 200], ["left"]),
        ("Too close", [120, 40, 400, 400], ["back"]),
        ("Too far", [270, 190, 100, 100], ["closer"]),
        ("Left and close", [50, 140, 400, 400], ["right", "back"]),
    ]
    
    for desc, bbox, keywords in test_cases:
        metrics = calculate_framing_metrics(bbox, 640, 480)
        cues = generate_directional_cues(metrics)
        
        print(f"  {desc}:")
        print(f"    Cues: '{cues}'")
        
        # Check if expected keywords are present
        matches = all(any(kw.lower() in cues.lower() for kw in [k]) for k in keywords)
        print(f"    Contains expected keywords: {matches}")
    print()


def test_well_framed():
    """Test well-framed detection"""
    print("Testing is_well_framed()...")
    
    test_cases = [
        # (description, bbox, expected_result)
        ("Perfect center, ideal size", [192, 96, 256, 288], True),
        ("Centered but too small", [270, 190, 100, 100], False),
        ("Centered but too large", [70, 0, 500, 480], False),
        ("Good size but off-center", [50, 140, 200, 200], False),
    ]
    
    for desc, bbox, expected in test_cases:
        metrics = calculate_framing_metrics(bbox, 640, 480)
        result = is_well_framed(metrics)
        
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: {result} (expected {expected})")
    print()


def test_main_function():
    """Test the main function with various scenarios"""
    print("Testing main() function...")
    
    # Reset lock before testing
    reset_lock()
    
    # Test 1: Object too far left
    print("\n  Scenario 1: Object too far left")
    image = create_test_image_with_object(obj_center_x=0.3, obj_center_y=0.5, obj_size=0.3)
    result = main(image, {'confidence': 0.3})
    print(f"    Result: '{result}'")
    
    # Test 2: Object centered and good size
    print("\n  Scenario 2: Well-framed object")
    image = create_test_image_with_object(obj_center_x=0.5, obj_center_y=0.5, obj_size=0.4)
    result = main(image, {'confidence': 0.3})
    print(f"    Result: '{result}'")
    
    # Test 3: Object too close
    print("\n  Scenario 3: Object too close")
    image = create_test_image_with_object(obj_center_x=0.5, obj_center_y=0.5, obj_size=0.8)
    result = main(image, {'confidence': 0.3})
    print(f"    Result: '{result}'")
    
    # Test 4: Object too far
    print("\n  Scenario 4: Object too far")
    image = create_test_image_with_object(obj_center_x=0.5, obj_center_y=0.5, obj_size=0.1)
    result = main(image, {'confidence': 0.3})
    print(f"    Result: '{result}'")
    
    # Test 5: No image
    print("\n  Scenario 5: No image")
    result = main(None)
    print(f"    Result: '{result}'")
    print()


def test_object_locking():
    """Test object locking and tracking"""
    print("Testing object locking...")
    
    # Reset lock
    reset_lock()
    
    # Create sequence of images with object moving
    print("\n  Frame 1: Object on left")
    image1 = create_test_image_with_object(obj_center_x=0.3, obj_center_y=0.5, obj_size=0.3)
    result1 = main(image1, {'confidence': 0.3, 'lock_object': True})
    print(f"    Result: '{result1}'")
    
    print("\n  Frame 2: Object moved slightly right")
    image2 = create_test_image_with_object(obj_center_x=0.4, obj_center_y=0.5, obj_size=0.3)
    result2 = main(image2, {'confidence': 0.3, 'lock_object': True})
    print(f"    Result: '{result2}'")
    
    print("\n  Frame 3: Object centered")
    image3 = create_test_image_with_object(obj_center_x=0.5, obj_center_y=0.5, obj_size=0.4)
    result3 = main(image3, {'confidence': 0.3, 'lock_object': True})
    print(f"    Result: '{result3}'")
    
    # Test reset
    print("\n  Testing reset...")
    result_reset = main(image1, {'reset': True})
    print(f"    Reset result: '{result_reset}'")
    print()


def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("Camera Aiming Assistance Tool - Test Suite")
    print("=" * 60)
    print()
    
    test_framing_metrics()
    test_directional_cues()
    test_well_framed()
    test_main_function()
    test_object_locking()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
