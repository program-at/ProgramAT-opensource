"""
Test script for empty_seat_detection.py
Tests the empty seat detection tool with sample data
"""

import sys
import os
import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from empty_seat_detection import (
    detect_objects,
    is_chair_occupied,
    calculate_iou,
    get_position_description,
    group_seats_by_location,
    generate_navigation_guidance,
    create_audio_description,
    main
)


def create_test_image_with_seats():
    """Create a test image with chairs and people"""
    # Create a blank image (white background)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some chair-like rectangles (brown)
    # Left side - 2 empty chairs
    cv2.rectangle(image, (50, 300), (100, 400), (40, 80, 100), -1)
    cv2.rectangle(image, (120, 300), (170, 400), (40, 80, 100), -1)
    
    # Center - 1 occupied chair with person
    cv2.rectangle(image, (295, 300), (345, 400), (40, 80, 100), -1)
    # Person sitting (skin tone)
    cv2.ellipse(image, (320, 250), (30, 40), 0, 0, 360, (180, 150, 120), -1)
    
    # Right side - 3 empty chairs
    cv2.rectangle(image, (450, 300), (500, 400), (40, 80, 100), -1)
    cv2.rectangle(image, (520, 300), (570, 400), (40, 80, 100), -1)
    cv2.rectangle(image, (450, 150), (500, 250), (40, 80, 100), -1)
    
    return image


def test_calculate_iou():
    """Test IoU calculation"""
    print("Testing calculate_iou()...")
    
    # Test cases: bbox1, bbox2, expected IoU (approximate)
    test_cases = [
        # Complete overlap
        ([100, 100, 50, 50], [100, 100, 50, 50], 1.0),
        # No overlap
        ([100, 100, 50, 50], [200, 200, 50, 50], 0.0),
        # Partial overlap
        ([100, 100, 50, 50], [125, 100, 50, 50], 0.5),
    ]
    
    for bbox1, bbox2, expected in test_cases:
        iou = calculate_iou(bbox1, bbox2)
        print(f"  IoU({bbox1}, {bbox2}) = {iou:.2f} (expected ~{expected:.2f})")
    print()


def test_is_chair_occupied():
    """Test chair occupancy detection"""
    print("Testing is_chair_occupied()...")
    
    # Empty chair
    chair = {
        'bbox': [100, 100, 50, 50],
        'center': [125, 125]
    }
    people = []
    result = is_chair_occupied(chair, people)
    print(f"  Empty chair (no people): {result} (expected False)")
    
    # Chair with person sitting on it
    chair = {
        'bbox': [100, 100, 50, 50],
        'center': [125, 125]
    }
    people = [{
        'bbox': [110, 80, 30, 60],  # Person overlapping chair
        'center': [125, 110]
    }]
    result = is_chair_occupied(chair, people)
    print(f"  Occupied chair (person sitting): {result} (expected True)")
    
    # Chair with person far away
    chair = {
        'bbox': [100, 100, 50, 50],
        'center': [125, 125]
    }
    people = [{
        'bbox': [300, 300, 30, 60],  # Person far away
        'center': [315, 330]
    }]
    result = is_chair_occupied(chair, people)
    print(f"  Empty chair (person far away): {result} (expected False)")
    print()


def test_position_description():
    """Test position descriptions"""
    print("Testing get_position_description()...")
    
    test_cases = [
        ([100, 400], "front left"),
        ([320, 240], "center"),
        ([540, 400], "front right"),
        ([100, 100], "back left"),
        ([320, 100], "back"),
    ]
    
    for center, expected_contains in test_cases:
        result = get_position_description(center, 640, 480)
        print(f"  Center {center} → '{result}'")
    print()


def test_group_seats_by_location():
    """Test grouping seats by location"""
    print("Testing group_seats_by_location()...")
    
    empty_seats = [
        {'center': [100, 400], 'class_name': 'chair'},
        {'center': [150, 400], 'class_name': 'chair'},
        {'center': [500, 400], 'class_name': 'chair'},
        {'center': [550, 400], 'class_name': 'chair'},
        {'center': [320, 240], 'class_name': 'chair'},
    ]
    
    grouped = group_seats_by_location(empty_seats, 640, 480)
    print(f"  Grouped seats:")
    for location, seats in grouped.items():
        print(f"    {location}: {len(seats)} seat(s)")
    print()


def test_navigation_guidance():
    """Test navigation guidance generation"""
    print("Testing generate_navigation_guidance()...")
    
    # Seats on the left
    empty_seats = [
        {'center': [100, 400], 'bbox': [75, 350, 50, 100], 'class_name': 'chair'},
        {'center': [150, 400], 'bbox': [125, 350, 50, 100], 'class_name': 'chair'},
    ]
    grouped = group_seats_by_location(empty_seats, 640, 480)
    
    guidance = generate_navigation_guidance(empty_seats, grouped, 640, 480)
    print(f"  Seats on left: '{guidance}'")
    
    # Seats on the right
    empty_seats = [
        {'center': [500, 400], 'bbox': [475, 350, 50, 100], 'class_name': 'chair'},
    ]
    grouped = group_seats_by_location(empty_seats, 640, 480)
    
    guidance = generate_navigation_guidance(empty_seats, grouped, 640, 480)
    print(f"  Seats on right: '{guidance}'")
    print()


def test_audio_description():
    """Test audio description generation"""
    print("Testing create_audio_description()...")
    
    # Scenario 1: Multiple empty seats in different locations
    empty_seats = [
        {'center': [100, 400], 'bbox': [75, 350, 50, 100], 'class_name': 'chair'},
        {'center': [150, 400], 'bbox': [125, 350, 50, 100], 'class_name': 'chair'},
        {'center': [500, 400], 'bbox': [475, 350, 50, 100], 'class_name': 'chair'},
        {'center': [550, 400], 'bbox': [525, 350, 50, 100], 'class_name': 'chair'},
        {'center': [500, 200], 'bbox': [475, 150, 50, 100], 'class_name': 'chair'},
    ]
    grouped = group_seats_by_location(empty_seats, 640, 480)
    
    description = create_audio_description(
        total_seats=6,
        occupied_seats=1,
        empty_seats=empty_seats,
        grouped_seats=grouped,
        width=640,
        height=480,
        include_navigation=True
    )
    print(f"  Multiple seats: '{description}'")
    print()
    
    # Scenario 2: One empty seat
    empty_seats = [
        {'center': [320, 240], 'bbox': [295, 190, 50, 100], 'class_name': 'chair'},
    ]
    grouped = group_seats_by_location(empty_seats, 640, 480)
    
    description = create_audio_description(
        total_seats=3,
        occupied_seats=2,
        empty_seats=empty_seats,
        grouped_seats=grouped,
        width=640,
        height=480,
        include_navigation=True
    )
    print(f"  Single seat: '{description}'")
    print()
    
    # Scenario 3: No empty seats
    description = create_audio_description(
        total_seats=3,
        occupied_seats=3,
        empty_seats=[],
        grouped_seats={},
        width=640,
        height=480,
        include_navigation=False
    )
    print(f"  No empty seats: '{description}'")
    print()


def test_main_function():
    """Test main entry point"""
    print("Testing main() function...")
    
    # Test with synthetic image
    image = create_test_image_with_seats()
    
    print("  Testing with sample image...")
    result = main(image)
    print(f"  Result: '{result}'")
    print()
    
    # Test with no image
    print("  Testing with None image...")
    result = main(None)
    print(f"  Result: '{result}'")
    print()
    
    # Test with config
    print("  Testing with config (no navigation)...")
    result = main(image, {'include_navigation': False})
    print(f"  Result: '{result}'")
    print()


def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("Empty Seat Detection Tool - Test Suite")
    print("=" * 60)
    print()
    
    test_calculate_iou()
    test_is_chair_occupied()
    test_position_description()
    test_group_seats_by_location()
    test_navigation_guidance()
    test_audio_description()
    test_main_function()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
