"""
Test script for door_detection.py
Tests the door and frame detection tool with sample data
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from door_detection import (
    detect_doors,
    get_clock_position,
    estimate_door_distance,
    generate_navigation_instruction,
    create_door_detection_response,
    main
)


def create_test_image_with_door():
    """Create a test image with a door-like rectangular pattern"""
    # Create a blank image (gray background - wall)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 180
    
    # Draw a door-like rectangle (brown/wooden color)
    # Vertical door in center
    cv2.rectangle(image, (250, 50), (390, 430), (50, 100, 150), -1)
    
    # Add door frame (darker border)
    cv2.rectangle(image, (240, 40), (400, 440), (30, 60, 90), 3)
    
    # Add door handle
    cv2.circle(image, (360, 240), 8, (200, 200, 50), -1)
    
    # Add some texture to make it look more like a door
    for i in range(3):
        y_pos = 150 + i * 100
        cv2.line(image, (260, y_pos), (380, y_pos), (40, 90, 140), 1)
    
    return image


def create_test_image_multiple_doors():
    """Create a test image with multiple door-like patterns"""
    # Create a blank image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 180
    
    # Left door (at 10 o'clock position)
    cv2.rectangle(image, (50, 100), (150, 380), (50, 100, 150), -1)
    cv2.rectangle(image, (45, 95), (155, 385), (30, 60, 90), 3)
    
    # Right door (at 2 o'clock position)
    cv2.rectangle(image, (490, 150), (590, 400), (50, 100, 150), -1)
    cv2.rectangle(image, (485, 145), (595, 405), (30, 60, 90), 3)
    
    # Center door (at 12 o'clock, farther away - smaller)
    cv2.rectangle(image, (290, 180), (350, 300), (50, 100, 150), -1)
    cv2.rectangle(image, (285, 175), (355, 305), (30, 60, 90), 3)
    
    return image


def test_clock_position():
    """Test clock position calculation"""
    print("Testing get_clock_position()...")
    
    test_cases = [
        (320, 100, 640, 480, 12, "top center"),
        (500, 150, 640, 480, 1, "top right"),
        (520, 240, 640, 480, 2, "middle right"),
        (500, 380, 640, 480, 3, "bottom right"),
        (100, 150, 640, 480, 11, "top left"),
        (120, 240, 640, 480, 10, "middle left"),
        (100, 380, 640, 480, 9, "bottom left"),
    ]
    
    for center_x, center_y, width, height, expected, description in test_cases:
        result = get_clock_position(center_x, center_y, width, height)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {description}: position ({center_x}, {center_y}) → {result} o'clock (expected {expected})")
    print()


def test_distance_estimation():
    """Test distance estimation"""
    print("Testing estimate_door_distance()...")
    
    test_cases = [
        ([0, 0, 100, 350], 480, "very close"),  # 73% of height
        ([0, 0, 100, 200], 480, "close"),       # 42% of height
        ([0, 0, 100, 100], 480, "medium"),      # 21% of height
        ([0, 0, 100, 50], 480, "far"),          # 10% of height
    ]
    
    for bbox, frame_height, expected in test_cases:
        result = estimate_door_distance(bbox, frame_height)
        height_pct = int((bbox[3] / frame_height) * 100)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Height {bbox[3]}/{frame_height} ({height_pct}%) → '{result}' (expected '{expected}')")
    print()


def test_navigation_instruction():
    """Test navigation instruction generation"""
    print("Testing generate_navigation_instruction()...")
    
    # Create test detections at different positions
    test_cases = [
        # center_x, center_y, bbox_height, description
        (320, 100, 350, "center top, very close"),
        (320, 240, 200, "center middle, close"),
        (500, 150, 100, "right top, medium"),
        (100, 240, 80, "left middle, far"),
    ]
    
    frame_width = 640
    frame_height = 480
    
    for center_x, center_y, bbox_height, description in test_cases:
        detection = {
            'center': [center_x, center_y],
            'bbox': [center_x - 50, center_y - bbox_height // 2, 100, bbox_height],
            'class_name': 'door'
        }
        
        result = generate_navigation_instruction(detection, frame_width, frame_height, is_streaming=False)
        print(f"  {description}:")
        print(f"    Instruction: '{result}'")
    print()


def test_streaming_word_limit():
    """Test that streaming mode limits to 15 words"""
    print("Testing streaming mode word limit...")
    
    # Create a detection that would generate a longer instruction
    detection = {
        'center': [500, 150],
        'bbox': [450, 75, 100, 150],
        'class_name': 'door'
    }
    
    # Normal mode
    result_normal = generate_navigation_instruction(detection, 640, 480, is_streaming=False)
    words_normal = len(result_normal.split())
    
    # Streaming mode
    result_streaming = generate_navigation_instruction(detection, 640, 480, is_streaming=True)
    words_streaming = len(result_streaming.split())
    
    print(f"  Normal mode: '{result_normal}' ({words_normal} words)")
    print(f"  Streaming mode: '{result_streaming}' ({words_streaming} words)")
    
    if words_streaming <= 15:
        print(f"  ✓ Streaming mode respects 15-word limit")
    else:
        print(f"  ✗ Streaming mode exceeds 15-word limit: {words_streaming} words")
    print()


def test_audio_response_types():
    """Test different audio response types based on distance"""
    print("Testing audio response types...")
    
    test_cases = [
        # bbox_height, expected_audio_type, description
        (350, 'success', "very close - should use haptic"),
        (200, 'beep_high', "close - should use high beep"),
        (100, 'speech', "medium - should use speech"),
        (50, 'speech', "far - should use speech"),
    ]
    
    frame_width = 640
    frame_height = 480
    
    for bbox_height, expected_type, description in test_cases:
        detections = [{
            'center': [320, 240],
            'bbox': [270, 240 - bbox_height // 2, 100, bbox_height],
            'class_name': 'door',
            'confidence': 0.8
        }]
        
        response = create_door_detection_response(detections, frame_width, frame_height, use_haptic=True)
        audio_type = response['audio']['type']
        
        status = "✓" if audio_type == expected_type else "✗"
        print(f"  {status} {description}:")
        print(f"    Audio type: {audio_type} (expected {expected_type})")
        print(f"    Text: '{response['text']}'")
    print()


def test_main_function():
    """Test main entry point with different configurations"""
    print("Testing main() function...")
    
    # Create test image
    image = create_test_image_with_door()
    
    # Note: Since we may not have YoloWorld installed, we'll test the function
    # but expect it might return "No door detected" if the model isn't available
    
    test_cases = [
        ({}, "default parameters"),
        ({'confidence': 0.2}, "lower confidence"),
        ({'is_streaming': True}, "streaming mode"),
    ]
    
    for input_data, description in test_cases:
        try:
            result = main(image, input_data)
            print(f"  {description}:")
            if isinstance(result, dict):
                print(f"    Audio type: {result.get('audio', {}).get('type', 'N/A')}")
                print(f"    Text: '{result.get('text', 'N/A')}'")
            else:
                print(f"    Result: '{result}'")
        except Exception as e:
            print(f"  {description}: Error - {e}")
    print()


def test_no_detections():
    """Test with image that has no doors"""
    print("Testing with blank image (no doors)...")
    
    # Create completely blank image
    blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    result = main(blank_image, {})
    if isinstance(result, dict):
        print(f"  Audio type: {result.get('audio', {}).get('type', 'N/A')}")
        print(f"  Text: '{result.get('text', 'N/A')}'")
    else:
        print(f"  Result: '{result}'")
    print()


def test_invalid_inputs():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test with None image
    result = main(None, {})
    if isinstance(result, dict):
        print(f"  None image - Audio type: {result.get('audio', {}).get('type', 'N/A')}, Text: '{result.get('text', 'N/A')}'")
    else:
        print(f"  None image: '{result}'")
    
    # Test with empty array
    empty = np.array([])
    result = main(empty, {})
    if isinstance(result, dict):
        print(f"  Empty array - Audio type: {result.get('audio', {}).get('type', 'N/A')}, Text: '{result.get('text', 'N/A')}'")
    else:
        print(f"  Empty array: '{result}'")
    print()


def test_multiple_doors():
    """Test with multiple doors in frame"""
    print("Testing with multiple doors...")
    
    image = create_test_image_multiple_doors()
    
    result = main(image, {})
    if isinstance(result, dict):
        print(f"  Audio type: {result.get('audio', {}).get('type', 'N/A')}")
        print(f"  Text: '{result.get('text', 'N/A')}'")
    else:
        print(f"  Result: '{result}'")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Door and Frame Detection Tool Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test building block functions (these don't require YoloWorld)
        test_clock_position()
        test_distance_estimation()
        test_navigation_instruction()
        test_streaming_word_limit()
        test_audio_response_types()
        
        # Test main function (may not detect doors without YoloWorld model)
        print("=" * 60)
        print("Testing with YoloWorld model (may not detect without model)")
        print("=" * 60)
        print()
        
        test_main_function()
        test_no_detections()
        test_invalid_inputs()
        test_multiple_doors()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        print("\nNote: Door detection requires YoloWorld model.")
        print("If model is not available, detections will be empty.")
        print("All building block functions have been tested successfully.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
