"""
Integration test for object_recognition tool with backend system
Tests that the tool can be loaded and executed in the streaming environment
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from object_recognition import main

def create_test_frame():
    """Create a test camera frame"""
    # Create a simple test image with a colored rectangle
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a red rectangle (simulating an object)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), -1)
    
    # Draw a face-like pattern
    cv2.ellipse(frame, (400, 200), (60, 80), 0, 0, 360, (180, 150, 120), -1)
    cv2.circle(frame, (380, 180), 8, (0, 0, 0), -1)
    cv2.circle(frame, (420, 180), 8, (0, 0, 0), -1)
    cv2.ellipse(frame, (400, 220), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    return frame

def test_basic_execution():
    """Test that the tool executes and returns audio-friendly output"""
    print("Test 1: Basic execution...")
    
    frame = create_test_frame()
    result = main(frame, {})
    
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"
    
    print(f"  ✓ Result: '{result}'")
    print()

def test_with_config():
    """Test with various configuration options"""
    print("Test 2: Configuration options...")
    
    frame = create_test_frame()
    
    configs = [
        {'confidence': 0.3},
        {'include_positions': False},
        {'include_distance': False},
        {'max_objects': 5},
    ]
    
    for config in configs:
        result = main(frame, config)
        assert isinstance(result, str), f"Result should be a string for config {config}"
        print(f"  ✓ Config {config}: '{result}'")
    
    print()

def test_streaming_mode():
    """Test tracking mode for streaming"""
    print("Test 3: Streaming/tracking mode...")
    
    frame = create_test_frame()
    
    # First frame with tracking enabled
    result1 = main(frame, {'track_mode': True})
    print(f"  ✓ First frame: '{result1}'")
    
    # Second frame (same objects, should return empty)
    result2 = main(frame, {'track_mode': True})
    print(f"  ✓ Second frame (no new objects): '{result2}'")
    
    # Create a new frame with additional object
    frame2 = create_test_frame()
    cv2.rectangle(frame2, (450, 350), (600, 450), (0, 255, 0), -1)
    
    result3 = main(frame2, {'track_mode': True})
    print(f"  ✓ Third frame (new object): '{result3}'")
    
    print()

def test_error_handling():
    """Test error cases"""
    print("Test 4: Error handling...")
    
    # None image
    result = main(None, {})
    assert result == "No camera image available"
    print(f"  ✓ None image: '{result}'")
    
    # Empty array
    empty = np.array([])
    result = main(empty, {})
    assert result == "No camera image available"
    print(f"  ✓ Empty array: '{result}'")
    
    # Invalid input_data
    frame = create_test_frame()
    result = main(frame, "some string")
    assert isinstance(result, str)
    print(f"  ✓ String input_data: '{result}'")
    
    print()

def test_audio_friendly_output():
    """Verify output is suitable for text-to-speech"""
    print("Test 5: Audio-friendly output verification...")
    
    frame = create_test_frame()
    result = main(frame, {})
    
    # Check for audio-friendly characteristics
    checks = [
        (not result.startswith('{'), "Should not start with JSON"),
        (not result.startswith('['), "Should not start with array"),
        ('object' in result.lower() or 'no objects' in result.lower(), "Should mention objects"),
    ]
    
    for condition, description in checks:
        assert condition, description
        print(f"  ✓ {description}")
    
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("Object Recognition Integration Test")
    print("=" * 60)
    print()
    
    try:
        test_basic_execution()
        test_with_config()
        test_streaming_mode()
        test_error_handling()
        test_audio_friendly_output()
        
        print("=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
