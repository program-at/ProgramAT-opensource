"""
Test script for object_recognition.py
Tests the object recognition building block with sample data
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from object_recognition import (
    detect_objects,
    count_objects_by_class,
    create_audio_description,
    get_position_description,
    estimate_distance,
    main
)

def create_test_image():
    """Create a simple test image with a colored rectangle"""
    # Create a blank image (white background)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a red rectangle (simulating an object)
    cv2.rectangle(image, (100, 100), (300, 300), (0, 0, 255), -1)
    
    # Draw a face-like pattern for Haar cascade detection
    # Face oval (skin tone)
    cv2.ellipse(image, (400, 200), (60, 80), 0, 0, 360, (180, 150, 120), -1)
    # Eyes
    cv2.circle(image, (380, 180), 8, (0, 0, 0), -1)
    cv2.circle(image, (420, 180), 8, (0, 0, 0), -1)
    # Mouth
    cv2.ellipse(image, (400, 220), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    return image

def test_detect_objects():
    """Test basic object detection"""
    print("Testing detect_objects()...")
    image = create_test_image()
    
    detections = detect_objects(image, confidence_threshold=0.5)
    print(f"  Found {len(detections)} objects")
    
    for i, det in enumerate(detections):
        print(f"  Object {i+1}: {det['class_name']} (confidence: {det['confidence']:.2f})")
    print()

def test_count_objects():
    """Test object counting"""
    print("Testing count_objects_by_class()...")
    image = create_test_image()
    
    detections = detect_objects(image)
    counts = count_objects_by_class(detections)
    
    print(f"  Object counts: {counts}")
    print()

def test_position_description():
    """Test position descriptions"""
    print("Testing get_position_description()...")
    
    test_cases = [
        ([100, 100], "top left"),
        ([320, 240], "center"),
        ([540, 400], "bottom right"),
    ]
    
    for center, expected in test_cases:
        result = get_position_description(center, 640, 480)
        print(f"  Center {center} → '{result}'")
    print()

def test_distance_estimation():
    """Test distance estimation"""
    print("Testing estimate_distance()...")
    
    test_cases = [
        ([0, 0, 100, 400], "very close"),  # Large object
        ([0, 0, 100, 200], "close"),
        ([0, 0, 100, 100], "medium distance"),
        ([0, 0, 100, 50], "far away"),
    ]
    
    for bbox, expected in test_cases:
        result = estimate_distance(bbox, 480)
        print(f"  Height {bbox[3]}/480 → '{result}'")
    print()

def test_audio_description():
    """Test audio description generation"""
    print("Testing create_audio_description()...")
    image = create_test_image()
    
    detections = detect_objects(image)
    height, width = image.shape[:2]
    
    description = create_audio_description(detections, width, height)
    print(f"  Audio description: '{description}'")
    print()

def test_main_function():
    """Test main entry point"""
    print("Testing main() function...")
    image = create_test_image()
    
    # Test with different input formats
    test_cases = [
        ({}, "default parameters"),
        ({'confidence': 0.3}, "lower confidence"),
        ({'include_positions': False}, "no positions"),
        ({'include_distance': False}, "no distance"),
        ("test string", "string input"),
    ]
    
    for input_data, description in test_cases:
        result = main(image, input_data)
        print(f"  {description}:")
        print(f"    '{result}'")
    print()

def test_empty_image():
    """Test with no detections"""
    print("Testing with blank image (no detections)...")
    # Create completely blank image
    blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    result = main(blank_image, {})
    print(f"  Result: '{result}'")
    print()

def test_invalid_inputs():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test with None image
    result = main(None, {})
    print(f"  None image: '{result}'")
    
    # Test with empty array
    empty = np.array([])
    result = main(empty, {})
    print(f"  Empty array: '{result}'")
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("Object Recognition Tool Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_detect_objects()
        test_count_objects()
        test_position_description()
        test_distance_estimation()
        test_audio_description()
        test_main_function()
        test_empty_image()
        test_invalid_inputs()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
