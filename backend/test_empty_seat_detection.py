"""Focused tests for the empty chair clockface tool behavior."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from empty_seat_detection import (
    calculate_iou,
    create_audio_description,
    get_clock_position,
    get_nearest_empty_chair,
    is_chair_occupied,
    limit_streaming_words,
    main,
)


class EmptySeatDetectionTests(unittest.TestCase):
    def test_calculate_iou(self):
        self.assertEqual(calculate_iou([100, 100, 50, 50], [200, 200, 50, 50]), 0.0)
        self.assertEqual(calculate_iou([100, 100, 50, 50], [100, 100, 50, 50]), 1.0)

    def test_is_chair_occupied_by_overlapping_person(self):
        chair = {'bbox': [100, 100, 50, 50], 'center': [125, 125]}
        people = [{'bbox': [110, 80, 30, 60], 'center': [125, 110]}]
        self.assertTrue(is_chair_occupied(chair, people))

    def test_clock_position_uses_visible_clockface_regions(self):
        self.assertEqual(get_clock_position(320, 100, 640, 480), 12)
        self.assertEqual(get_clock_position(100, 150, 640, 480), 11)
        self.assertEqual(get_clock_position(520, 240, 640, 480), 2)

    def test_nearest_empty_chair_prefers_lower_larger_detection(self):
        empty_chairs = [
            {'center': [100, 250], 'bbox': [70, 200, 60, 100], 'class_name': 'chair'},
            {'center': [500, 360], 'bbox': [450, 290, 100, 140], 'class_name': 'chair'},
        ]
        nearest = get_nearest_empty_chair(empty_chairs)
        self.assertEqual(nearest, empty_chairs[1])

    def test_audio_description_reports_nearest_clockface_chair(self):
        empty_chairs = [
            {'center': [120, 170], 'bbox': [80, 120, 80, 100], 'class_name': 'chair'},
            {'center': [120, 390], 'bbox': [70, 300, 100, 160], 'class_name': 'chair'},
        ]
        description = create_audio_description(
            total_seats=2,
            occupied_seats=0,
            empty_seats=empty_chairs,
            grouped_seats={},
            width=640,
            height=480,
            include_navigation=True,
        )
        self.assertEqual(description, "Two empty chairs, nearest at nine o'clock.")

    def test_audio_description_handles_no_empty_chairs(self):
        description = create_audio_description(
            total_seats=2,
            occupied_seats=2,
            empty_seats=[],
            grouped_seats={},
            width=640,
            height=480,
            include_navigation=False,
        )
        self.assertEqual(description, "No empty chairs in view.")

    def test_streaming_word_limit_caps_output(self):
        message = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
        self.assertEqual(len(limit_streaming_words(message).split()), 15)

    def test_main_reports_empty_chair_clockface(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {'class_name': 'chair', 'bbox': [70, 300, 100, 160], 'center': [120, 380]},
            {'class_name': 'chair', 'bbox': [450, 140, 80, 100], 'center': [490, 190]},
            {'class_name': 'person', 'bbox': [460, 120, 90, 140], 'center': [505, 190]},
        ]

        # Mock detections so the test stays deterministic and does not depend on YOLO downloads.
        with patch('empty_seat_detection.detect_objects', return_value=detections):
            result = main(image)

        self.assertEqual(result, "One empty chair at nine o'clock.")

    def test_main_streaming_suppresses_repeated_state(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {'class_name': 'chair', 'bbox': [250, 220, 120, 180], 'center': [310, 310]},
        ]

        # Mock detections so the test exercises streaming state without model side effects.
        with patch('empty_seat_detection.detect_objects', return_value=detections):
            first = main(image, {'is_streaming': True})
            second = main(image, {'is_streaming': True})

        self.assertEqual(first, "One empty chair at twelve o'clock.")
        self.assertEqual(second, "")


if __name__ == "__main__":
    unittest.main()
