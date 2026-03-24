"""
Test script for package_finding.py
Tests the package finding tool with synthetic test data
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from package_finding import (
    detect_packages,
    get_position_description,
    estimate_distance,
    get_size_description,
    get_directional_guidance,
    create_still_frame_response,
    create_streaming_response,
    main
)

def create_test_image_with_box():
    """Create a simple test image with a box-like shape"""
    # Create a blank image (white background)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a brown rectangle (simulating a cardboard box)
    # Large box in center-left
    cv2.rectangle(image, (100, 150), (350, 380), (139, 69, 19), -1)
    
    # Add some shading to make it look 3D
    cv2.rectangle(image, (100, 150), (350, 160), (100, 50, 10), -1)  # Top edge
    cv2.rectangle(image, (340, 150), (350, 380), (100, 50, 10), -1)  # Side edge
    
    return image

def create_test_image_no_packages():
    """Create a test image without any packages"""
    # Create a blank image (gray background)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw some non-package objects (just shapes)
    cv2.circle(image, (320, 240), 50, (100, 100, 255), -1)  # Blue circle
    
    return image

def test_position_description():
    """Test position descriptions"""
    print("Testing get_position_description()...")
    
    test_cases = [
        ([100, 100], 640, 480, "top left"),
        ([320, 240], 640, 480, "center"),
        ([540, 400], 640, 480, "bottom right"),
        ([100, 240], 640, 480, "left"),
    ]
    
    for center, width, height, expected in test_cases:
        result = get_position_description(center, width, height)
        match = "✓" if result == expected else "✗"
        print(f"  {match} Center {center} → '{result}' (expected: '{expected}')")
    print()
    print()

def test_distance_estimation():
    """Test distance estimation"""
    print("Testing estimate_distance()...")
    
    test_cases = [
        ([0, 0, 100, 400], 480, "very close"),  # Large object
        ([0, 0, 100, 200], 480, "close"),
        ([0, 0, 100, 100], 480, "medium distance"),
        ([0, 0, 100, 50], 480, "far away"),
    ]
    
    for bbox, frame_height, expected in test_cases:
        result = estimate_distance(bbox, frame_height)
        print(f"  BBox height {bbox[3]}/{frame_height} → '{result}' (expected: {expected})")
    print()

def test_size_description():
    """Test size descriptions"""
    print("Testing get_size_description()...")
    
    test_cases = [
        ([0, 0, 400, 400], 480, 640, "large"),
        ([0, 0, 200, 150], 480, 640, "medium"),
        ([0, 0, 50, 50], 480, 640, "small"),
    ]
    
    for bbox, height, width, expected in test_cases:
        result = get_size_description(bbox, height, width)
        print(f"  BBox {bbox[2]}x{bbox[3]} in {width}x{height} → '{result}' (expected: {expected})")
    print()

def test_directional_guidance():
    """Test directional guidance"""
    print("Testing get_directional_guidance()...")
    
    test_cases = [
        ([320, 240], 640, 480, "close", "straight ahead"),
        ([500, 240], 640, 480, "medium distance", "right"),
        ([100, 240], 640, 480, "far away", "left"),
    ]
    
    for center, width, height, distance, expected_dir in test_cases:
        result = get_directional_guidance(center, width, height, distance)
        print(f"  Center {center}, distance={distance}")
        print(f"    → '{result}'")
    print()

def test_still_frame_response():
    """Test still frame response generation"""
    print("Testing create_still_frame_response()...")
    
    # Test with no packages
    packages = []
    result = create_still_frame_response(packages, 640, 480)
    print(f"  No packages: '{result}'")
    
    # Test with one package
    packages = [{
        'class_name': 'box',
        'bbox': [100, 150, 250, 230],
        'center': [225, 265],
        'confidence': 0.85
    }]
    result = create_still_frame_response(packages, 640, 480)
    print(f"  One package: '{result}'")
    
    # Test with multiple packages
    packages = [
        {'class_name': 'box', 'bbox': [100, 150, 250, 230], 'center': [225, 265], 'confidence': 0.85},
        {'class_name': 'package', 'bbox': [400, 200, 100, 80], 'center': [450, 240], 'confidence': 0.75}
    ]
    result = create_still_frame_response(packages, 640, 480)
    print(f"  Multiple packages: '{result}'")
    print()

def test_streaming_response():
    """Test streaming response generation"""
    print("Testing create_streaming_response()...")
    
    # Reset state
    import package_finding
    package_finding._last_detection = None
    package_finding._last_distance_category = None
    
    # Test with no packages
    packages = []
    result = create_streaming_response(packages, 640, 480)
    print(f"  No packages: '{result}'")
    
    # Test with new package detection
    packages = [{
        'class_name': 'box',
        'bbox': [100, 150, 250, 230],
        'center': [225, 265],
        'confidence': 0.85
    }]
    result = create_streaming_response(packages, 640, 480)
    print(f"  New package: '{result}'")
    
    # Test with same package (should return empty)
    result = create_streaming_response(packages, 640, 480)
    print(f"  Same package: '{result}' (should be empty)")
    
    # Reset state
    package_finding._last_detection = None
    package_finding._last_distance_category = None
    print()

def test_main_still_mode():
    """Test main() in still mode"""
    print("Testing main() in still mode...")
    
    # Create test image
    image = create_test_image_with_box()
    
    # Note: This will try to use YoloWorld which may not be available
    # The test will show what happens in that case
    try:
        result = main(image, {'mode': 'still', 'confidence': 0.3})
        print(f"  Result: '{result}'")
    except Exception as e:
        print(f"  Note: YoloWorld may not be available in test environment")
        print(f"  Error: {e}")
    
    # Test with no camera
    result = main(None)
    print(f"  No image: '{result}'")
    print()

def test_main_stream_mode():
    """Test main() in stream mode"""
    print("Testing main() in stream mode...")
    
    # Create test image
    image = create_test_image_with_box()
    
    # Note: This will try to use YoloWorld which may not be available
    try:
        result = main(image, {'mode': 'stream', 'confidence': 0.3})
        print(f"  Stream result: '{result}'")
    except Exception as e:
        print(f"  Note: YoloWorld may not be available in test environment")
        print(f"  Error: {e}")
    print()

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Package Finding Tool Tests")
    print("=" * 60)
    print()
    
    # Test building blocks
    test_position_description()
    test_distance_estimation()
    test_size_description()
    test_directional_guidance()
    test_still_frame_response()
    test_streaming_response()
    
    # Test main functions
    test_main_still_mode()
    test_main_stream_mode()
    
    print("=" * 60)
    print("Tests completed!")
    print()
    print("Note: YoloWorld model tests require ultralytics and may")
    print("download the model on first run. The building block tests")
    print("validate the core logic independently.")
    print("=" * 60)

if __name__ == '__main__':
    run_all_tests()
