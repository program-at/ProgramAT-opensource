"""
Test script for physical_interface_aid.py
Tests the physical interface accessibility aid tool with sample data.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

# Try to import the tool module; some tests skip if dependencies are missing
try:
    import physical_interface_aid
    _MODULE_AVAILABLE = True
    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:
    _MODULE_AVAILABLE = False
    _IMPORT_ERROR = str(exc)


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


def _make_llm_result(response, artifact=None):
    """Helper: return a minimal copilot_llm_call result dict."""
    return {"response": response, "artifact": artifact or {}}


class TestPhysicalInterfaceAidImport(unittest.TestCase):
    """Tests that the tool module is importable and has the expected structure."""

    def test_module_importable(self):
        """The tool module must be importable without errors."""
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable (missing deps): {_IMPORT_ERROR}")

    def test_main_function_exists(self):
        """main() must exist and accept two parameters."""
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")
        import inspect
        self.assertTrue(hasattr(physical_interface_aid, 'main'))
        sig = inspect.signature(physical_interface_aid.main)
        self.assertEqual(len(sig.parameters), 2)

    def test_main_parameter_names(self):
        """main() must accept 'image' and 'input_data' parameters."""
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")
        import inspect
        sig = inspect.signature(physical_interface_aid.main)
        params = list(sig.parameters.keys())
        self.assertIn('image', params)
        self.assertIn('input_data', params)


class TestPhysicalInterfaceAidNoneImage(unittest.TestCase):
    """Tests tool behaviour when no image is provided."""

    def setUp(self):
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")

    def test_none_image_returns_error_dict(self):
        """Passing None for image should return an error dict, not raise."""
        result = physical_interface_aid.main(None, {})
        self.assertIsInstance(result, dict)
        self.assertIn('audio', result)
        self.assertEqual(result['audio']['type'], 'error')

    def test_empty_array_returns_error_dict(self):
        """Passing an empty numpy array should return an error dict, not raise."""
        result = physical_interface_aid.main(np.array([]), {})
        self.assertIsInstance(result, dict)
        self.assertIn('audio', result)
        self.assertEqual(result['audio']['type'], 'error')


class TestPhysicalInterfaceAidWithMock(unittest.TestCase):
    """
    Deterministic tests using a mocked copilot_llm_call so they run
    reliably in any environment without requiring a live backend.
    """

    def setUp(self):
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")
        # Reset streaming state before each test
        physical_interface_aid._last_response = ""
        physical_interface_aid._frame_count = 0

    def _patched_llm(self, responses):
        """Return a side_effect list for copilot_llm_call that yields each response dict."""
        return [_make_llm_result(r) for r in responses]

    def test_returns_str_on_success(self):
        """main() returns a non-empty string when all stages succeed."""
        side_effects = self._patched_llm([
            "microwave keypad|buttons: 1(top-left), Start(bottom-right)|finger: center",
            "finger is near the 5 button; move right to Start",
            "Move right to the Start button",
        ])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Move right to the Start button")

    def test_streaming_deduplication_returns_empty_on_repeat(self):
        """Calling main() twice with the same scene should return '' on second call."""
        side_effects = self._patched_llm([
            "microwave|buttons: Start|finger: left",
            "finger left of Start; move right",
            "Move right to the Start button",
            # Second invocation
            "microwave|buttons: Start|finger: left",
            "finger left of Start; move right",
            "Move right to the Start button",
        ])
        image = create_blank_image()
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            first = physical_interface_aid.main(image, {})
            second = physical_interface_aid.main(image, {})
        self.assertEqual(first, "Move right to the Start button")
        self.assertEqual(second, "")

    def test_new_response_after_scene_change_is_spoken(self):
        """A different guidance should not be suppressed by deduplication."""
        side_effects = self._patched_llm([
            "microwave|buttons: Start|finger: center",
            "finger on Start",
            "Your finger is on the Start button",
            # Second invocation with different result
            "microwave|buttons: Start|finger: right",
            "finger right of Start; move left",
            "Move left to the Start button",
        ])
        image = create_blank_image()
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            first = physical_interface_aid.main(image, {})
            second = physical_interface_aid.main(image, {})
        self.assertNotEqual(first, "")
        self.assertNotEqual(second, "")
        self.assertNotEqual(first, second)

    def test_response_word_count_at_most_15(self):
        """Guidance responses must be at most 15 words.

        Note: this validates that the tool correctly passes through a
        ≤15-word response returned by the mocked LLM.  It does not
        guarantee that a live LLM will honour the 15-word system-prompt
        instruction, which is tested separately in integration.
        """
        guidance = "Move right to the Start button"
        side_effects = self._patched_llm([
            "interface", "spatial", guidance,
        ])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        if isinstance(result, str) and result:
            self.assertLessEqual(len(result.split()), 15,
                                 f"Response exceeds 15 words: '{result}'")

    def test_frame_count_increments(self):
        """_frame_count must increment on each main() call."""
        physical_interface_aid._frame_count = 0
        side_effects = self._patched_llm(["interface", "spatial", "Move up to the 7 button"])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            physical_interface_aid.main(create_blank_image(), {})
        self.assertGreaterEqual(physical_interface_aid._frame_count, 1)

    def test_empty_guidance_response_returns_empty_string(self):
        """When the navigation stage returns an empty response, return ''."""
        side_effects = self._patched_llm(["interface", "spatial", ""])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertEqual(result, "")

    def test_exception_returns_error_dict(self):
        """An exception from copilot_llm_call must return an error dict, not raise."""
        with patch.object(physical_interface_aid, 'copilot_llm_call',
                          side_effect=RuntimeError("backend unavailable")):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertIsInstance(result, dict)
        self.assertIn('audio', result)
        self.assertEqual(result['audio']['type'], 'error')
        self.assertIn("backend unavailable", result['audio']['text'])

    def test_error_message_word_boundary_truncation(self):
        """Long error messages should be truncated at a word boundary with ellipsis."""
        long_error = "this is a very long error message " * 10  # 340+ chars
        with patch.object(physical_interface_aid, 'copilot_llm_call',
                          side_effect=RuntimeError(long_error)):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertIsInstance(result, dict)
        # Raw portion after prefix must end with '…' and be ≤151 chars.
        # The tool slices at 150 chars then rsplit may reduce that further,
        # giving content ≤150 chars; adding '…' makes the maximum total 151.
        raw_in_text = result['audio']['text'].replace("Interface navigation error: ", "")
        self.assertTrue(raw_in_text.endswith("…"),
                        f"Expected truncated message to end with '…', got: {raw_in_text!r}")
        self.assertLessEqual(len(raw_in_text), 151,
                             f"Truncated message too long ({len(raw_in_text)} chars)")


class TestPhysicalInterfaceAidReturnTypes(unittest.TestCase):
    """Tests that return values conform to the tool contract (mocked)."""

    def setUp(self):
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")
        physical_interface_aid._last_response = ""
        physical_interface_aid._frame_count = 0

    def test_dict_result_has_audio_key(self):
        """If main() returns a dict, it must include an 'audio' key."""
        # Force an error path via bad image
        result = physical_interface_aid.main(None, {})
        if isinstance(result, dict):
            self.assertIn('audio', result)

    def test_dict_audio_has_required_keys(self):
        """audio sub-dict must have at least 'type' and 'text' keys."""
        result = physical_interface_aid.main(None, {})
        if isinstance(result, dict) and 'audio' in result:
            audio = result['audio']
            self.assertIn('type', audio)
            self.assertIn('text', audio)

    def test_audio_type_is_valid(self):
        """audio.type must be one of the recognised audio types."""
        valid_types = {'speech', 'beep_high', 'beep_low', 'success', 'warning', 'error'}
        result = physical_interface_aid.main(None, {})
        if isinstance(result, dict) and 'audio' in result:
            self.assertIn(result['audio']['type'], valid_types)


class TestStreamingDeduplication(unittest.TestCase):
    """Tests the streaming state variables exist on the module."""

    def setUp(self):
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")

    def test_module_has_last_response_state(self):
        """Module must expose _last_response for streaming deduplication."""
        self.assertTrue(hasattr(physical_interface_aid, '_last_response'))

    def test_module_has_frame_count_state(self):
        """Module must expose _frame_count."""
        self.assertTrue(hasattr(physical_interface_aid, '_frame_count'))


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
            "Guardrail violations found:\n" + "\n".join(failures),
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
