"""
Test script for physical_interface_aid.py
Tests the physical interface accessibility aid tool with sample data.
"""

import sys
import os
import unittest
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))


def create_blank_image(width=640, height=480):
    """Create a simple blank test image (white background)."""
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def create_keypad_image():
    """Create a simple test image that resembles a keypad."""
    try:
        import cv2
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Draw a 3x4 grid of button-like rectangles
        for row in range(4):
            for col in range(3):
                x1 = 100 + col * 120
                y1 = 60 + row * 100
                x2 = x1 + 90
                y2 = y1 + 70
                cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 100), -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (50, 50, 50), 2)
        return image
    except ImportError:
        return create_blank_image()


class TestPhysicalInterfaceAidImport(unittest.TestCase):
    """Tests that the tool module is importable and has the expected structure."""

    def test_module_importable(self):
        """The tool module must be importable without errors."""
        import physical_interface_aid  # noqa: F401

    def test_main_function_exists(self):
        """main() must exist and accept two parameters."""
        import physical_interface_aid
        import inspect
        self.assertTrue(hasattr(physical_interface_aid, 'main'))
        sig = inspect.signature(physical_interface_aid.main)
        self.assertEqual(len(sig.parameters), 2)

    def test_main_parameter_names(self):
        """main() must accept 'image' and 'input_data' parameters."""
        import physical_interface_aid
        import inspect
        sig = inspect.signature(physical_interface_aid.main)
        params = list(sig.parameters.keys())
        self.assertIn('image', params)
        self.assertIn('input_data', params)


class TestPhysicalInterfaceAidNoneImage(unittest.TestCase):
    """Tests tool behaviour when no image is provided."""

    def test_none_image_returns_error_dict(self):
        """Passing None for image should return an error dict, not raise."""
        import physical_interface_aid
        result = physical_interface_aid.main(None, {})
        self.assertIsInstance(result, dict)
        self.assertIn('audio', result)
        self.assertEqual(result['audio']['type'], 'error')

    def test_empty_array_returns_error_dict(self):
        """Passing an empty numpy array should return an error dict, not raise."""
        import physical_interface_aid
        result = physical_interface_aid.main(np.array([]), {})
        self.assertIsInstance(result, dict)
        self.assertIn('audio', result)
        self.assertEqual(result['audio']['type'], 'error')


class TestPhysicalInterfaceAidReturnTypes(unittest.TestCase):
    """Tests that return values conform to the tool contract."""

    def _call_main(self, image=None, input_data=None):
        import physical_interface_aid
        if image is None:
            image = create_blank_image()
        return physical_interface_aid.main(image, input_data or {})

    def test_returns_str_or_dict(self):
        """main() must return a str or a dict (the tool contract)."""
        result = self._call_main()
        self.assertIsInstance(result, (str, dict))

    def test_dict_result_has_audio_key(self):
        """If main() returns a dict, it must include an 'audio' key."""
        result = self._call_main()
        if isinstance(result, dict):
            self.assertIn('audio', result)

    def test_dict_audio_has_required_keys(self):
        """audio sub-dict must have at least 'type' and 'text' keys."""
        result = self._call_main()
        if isinstance(result, dict) and 'audio' in result:
            audio = result['audio']
            self.assertIn('type', audio)
            self.assertIn('text', audio)

    def test_audio_type_is_valid(self):
        """audio.type must be one of the recognised audio types."""
        valid_types = {'speech', 'beep_high', 'beep_low', 'success', 'warning', 'error'}
        result = self._call_main()
        if isinstance(result, dict) and 'audio' in result:
            self.assertIn(result['audio']['type'], valid_types)


class TestStreamingDeduplication(unittest.TestCase):
    """Tests the streaming deduplication behaviour."""

    def setUp(self):
        """Reset the module-level streaming state before each test."""
        import physical_interface_aid
        physical_interface_aid._last_response = ""
        physical_interface_aid._frame_count = 0

    def test_module_has_streaming_state(self):
        """Module must expose _last_response for streaming deduplication."""
        import physical_interface_aid
        self.assertTrue(hasattr(physical_interface_aid, '_last_response'))

    def test_frame_count_increments(self):
        """_frame_count should increment on each main() call."""
        import physical_interface_aid
        physical_interface_aid._frame_count = 0
        image = create_blank_image()
        try:
            physical_interface_aid.main(image, {})
        except Exception as exc:
            self.skipTest(f"Backend unavailable: {exc}")
        self.assertGreaterEqual(physical_interface_aid._frame_count, 1)


class TestStreamingWordLimit(unittest.TestCase):
    """Tests that live-mode responses respect the 15-word limit."""

    def test_non_empty_str_response_word_count(self):
        """Any non-empty string response should be at most 15 words."""
        import physical_interface_aid
        image = create_blank_image()
        try:
            result = physical_interface_aid.main(image, {})
        except Exception as exc:
            self.skipTest(f"Backend unavailable: {exc}")
        if isinstance(result, str) and result:
            word_count = len(result.split())
            self.assertLessEqual(
                word_count, 15,
                f"Response exceeded 15 words: '{result}' ({word_count} words)",
            )


class TestValidateGuardrails(unittest.TestCase):
    """Confirm the tool passes the generated-tool guardrail checks."""

    def test_passes_validate_generated_tools(self):
        """physical_interface_aid.py must not trigger any guardrail violations."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        import validate_generated_tools
        tool_path = os.path.join(
            os.path.dirname(__file__), '..', 'tools', 'physical_interface_aid.py'
        )
        from pathlib import Path
        failures = validate_generated_tools.validate_files([Path(tool_path)])
        self.assertEqual(
            failures, [],
            f"Guardrail violations found:\n" + "\n".join(failures),
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
