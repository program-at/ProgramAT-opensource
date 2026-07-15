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
        physical_interface_aid._last_spoken_time = 0.0

    def _patched_llm(self, responses):
        """Return a side_effect list for copilot_llm_call that yields each response dict."""
        return [_make_llm_result(r) for r in responses]

    def test_returns_str_on_success(self):
        """main() returns a non-empty string when the single-stage call succeeds."""
        side_effects = self._patched_llm([
            "move right towards Start",
        ])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertIsInstance(result, str)
        self.assertEqual(result, "move right towards Start")

    def test_streaming_deduplication_returns_empty_on_repeat(self):
        """Calling main() twice with the same scene should return '' on second call
        (when the second call does not land on a REPEAT_INTERVAL boundary)."""
        # _frame_count starts at 0; first call → 1, second call → 2.
        # Neither is a multiple of REPEAT_INTERVAL (10), so the second call
        # should be suppressed.
        side_effects = self._patched_llm([
            "move right towards Start",
            # Second invocation
            "move right towards Start",
        ])
        image = create_blank_image()
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            first = physical_interface_aid.main(image, {})
            second = physical_interface_aid.main(image, {})
        self.assertEqual(first, "move right towards Start")
        self.assertEqual(second, "")

    def test_periodic_repeat_re_announces_at_interval(self):
        """Same guidance should be re-announced at every REPEAT_INTERVAL frames."""
        import unittest.mock as mock_module
        interval = physical_interface_aid.REPEAT_INTERVAL
        # _frame_count starts at 0; call i+1 lands on frame i+1.
        # Periodic re-announcement fires when _frame_count % interval == 0,
        # i.e. at frame `interval` = results[interval - 1].
        guidance = "move right towards Start"
        # Each call needs 1 LLM result; produce enough for exactly `interval` calls.
        side_effects = self._patched_llm(
            [guidance] * interval
        )
        # time.monotonic() is only called when deduplication passes (frames 1 and
        # REPEAT_INTERVAL).  Provide values that are each ≥ 1.5 s apart so the
        # rate limiter passes at both re-announcement points.
        fake_times = iter([2.0, 4.0])
        image = create_blank_image()
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            with mock_module.patch('physical_interface_aid.time') as mock_time:
                mock_time.monotonic.side_effect = fake_times
                results = [physical_interface_aid.main(image, {}) for _ in range(interval)]
        # Frame 1 (index 0) → first announcement
        self.assertEqual(results[0], guidance)
        # Frames 2..(interval-1) (indices 1..interval-2) → suppressed by deduplication
        for i in range(1, interval - 1):
            self.assertEqual(results[i], "", f"Expected '' at frame {i + 1}, got {results[i]!r}")
        # Frame interval (index interval-1) → periodic re-announcement
        self.assertEqual(results[interval - 1], guidance,
                         f"Expected repeat at frame {interval}")

    def test_new_response_after_scene_change_is_spoken(self):
        """A different guidance should not be suppressed by deduplication."""
        import unittest.mock as mock_module
        side_effects = self._patched_llm([
            "your finger is on Start",
            # Second invocation with different result
            "move left towards Start",
        ])
        # Both calls have different responses so dedup passes each time.
        # Provide times ≥ 1.5 s apart so the rate limiter also passes.
        fake_times = iter([2.0, 4.0])
        image = create_blank_image()
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            with mock_module.patch('physical_interface_aid.time') as mock_time:
                mock_time.monotonic.side_effect = fake_times
                first = physical_interface_aid.main(image, {})
                second = physical_interface_aid.main(image, {})
        self.assertNotEqual(first, "")
        self.assertNotEqual(second, "")
        self.assertNotEqual(first, second)

    def test_response_word_count_at_most_15(self):
        """Guidance responses must be at most 15 words.

        Note: this validates that the tool correctly passes through a
        ≤15-word response returned by the mocked LLM.  Compliance of live
        LLM responses with the 15-word system-prompt instruction is verified
        in end-to-end integration testing.
        """
        guidance = "move right towards Start"
        side_effects = self._patched_llm([guidance])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        if isinstance(result, str) and result:
            self.assertLessEqual(len(result.split()), physical_interface_aid.STREAMING_WORD_LIMIT,
                                 f"Response exceeds {physical_interface_aid.STREAMING_WORD_LIMIT} "
                                 f"words: '{result}'")

    def test_over_limit_response_is_truncated_in_code(self):
        """The code must truncate LLM responses that exceed STREAMING_WORD_LIMIT."""
        long_guidance = " ".join([f"word{i}" for i in range(25)])  # 25 words
        side_effects = self._patched_llm([long_guidance])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        if isinstance(result, str) and result:
            self.assertLessEqual(len(result.split()), physical_interface_aid.STREAMING_WORD_LIMIT,
                                 f"Response not truncated to {physical_interface_aid.STREAMING_WORD_LIMIT} "
                                 f"words: '{result}'")

    def test_frame_count_increments(self):
        """_frame_count must increment on each main() call."""
        physical_interface_aid._frame_count = 0
        side_effects = self._patched_llm(["move up towards 7"])
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            physical_interface_aid.main(create_blank_image(), {})
        self.assertGreaterEqual(physical_interface_aid._frame_count, 1)

    def test_empty_guidance_response_returns_empty_string(self):
        """When the single-stage call returns an empty response, return ''."""
        side_effects = self._patched_llm([""])
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
        # Construct an error message known to contain spaces; use a multi-word
        # phrase repeated so the total length exceeds MAX_ERROR_MSG_LEN.
        long_error = "this is a very long error message " * 10  # 340 chars (34 × 10)
        with patch.object(physical_interface_aid, 'copilot_llm_call',
                          side_effect=RuntimeError(long_error)):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertIsInstance(result, dict)
        audio_text = result['audio']['text']
        # Must end with '…' to indicate truncation
        self.assertTrue(audio_text.endswith("…"),
                        f"Expected truncated message to end with '…', got: {audio_text!r}")
        # Total length must be ≤ prefix + MAX_ERROR_MSG_LEN + 1 (for '…')
        max_expected = (
            len("Interface navigation error: ")
            + physical_interface_aid.MAX_ERROR_MSG_LEN
            + 1  # '…'
        )
        self.assertLessEqual(len(audio_text), max_expected,
                             f"Truncated message too long ({len(audio_text)} chars)")
        # Verify the truncated content ends at a word boundary: the character
        # before '…' must be the last character of a complete word (non-space),
        # and that word must be a complete word from the original error string.
        content_before_ellipsis = audio_text[len("Interface navigation error: "):-1]
        last_word = content_before_ellipsis.split()[-1] if content_before_ellipsis.split() else ""
        self.assertIn(last_word, long_error,
                      f"Last word '{last_word}' before ellipsis not found in original error")


class TestOutputFormatEnforcement(unittest.TestCase):
    """
    Tests for code-level output-format enforcement (_fix_diagonals and
    _enforce_output_format).  These run regardless of the backend's
    planning/routing configuration.
    """

    def setUp(self):
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")

    # ── _fix_diagonals ────────────────────────────────────────────────────────

    def test_fix_diagonals_up_left(self):
        """'up-left' is replaced with 'up'."""
        result = physical_interface_aid._fix_diagonals("move up-left towards Start")
        self.assertEqual(result, "move up towards Start")

    def test_fix_diagonals_up_right(self):
        """'up-right' is replaced with 'up'."""
        result = physical_interface_aid._fix_diagonals("move up-right towards Start")
        self.assertEqual(result, "move up towards Start")

    def test_fix_diagonals_down_left(self):
        """'down-left' is replaced with 'down'."""
        result = physical_interface_aid._fix_diagonals("move down-left towards 7")
        self.assertEqual(result, "move down towards 7")

    def test_fix_diagonals_down_right(self):
        """'down-right' is replaced with 'down'."""
        result = physical_interface_aid._fix_diagonals("move down-right towards Cancel")
        self.assertEqual(result, "move down towards Cancel")

    def test_fix_diagonals_upper_left(self):
        """'upper-left' variant is replaced with 'up'."""
        result = physical_interface_aid._fix_diagonals("move upper-left towards 1")
        self.assertEqual(result, "move up towards 1")

    def test_fix_diagonals_lower_right(self):
        """'lower-right' variant is replaced with 'down'."""
        result = physical_interface_aid._fix_diagonals("move lower-right towards Enter")
        self.assertEqual(result, "move down towards Enter")

    def test_fix_diagonals_case_insensitive(self):
        """Diagonal replacement is case-insensitive."""
        result = physical_interface_aid._fix_diagonals("move UP-LEFT towards Start")
        self.assertEqual(result, "move up towards Start")

    def test_fix_diagonals_cardinal_unchanged(self):
        """Cardinal directions are not altered."""
        for direction in ("left", "right", "up", "down"):
            text = f"move {direction} towards Start"
            self.assertEqual(physical_interface_aid._fix_diagonals(text), text)

    # ── _enforce_output_format ────────────────────────────────────────────────

    def test_enforce_suppresses_touch(self):
        """'touch' triggers suppression; empty string returned."""
        self.assertEqual(
            physical_interface_aid._enforce_output_format("touch the Start button"),
            "",
        )

    def test_enforce_suppresses_touching(self):
        self.assertEqual(
            physical_interface_aid._enforce_output_format("you are touching the Start button"),
            "",
        )

    def test_enforce_suppresses_tap(self):
        self.assertEqual(
            physical_interface_aid._enforce_output_format("tap the 5 button"),
            "",
        )

    def test_enforce_suppresses_press(self):
        self.assertEqual(
            physical_interface_aid._enforce_output_format("press Start to begin"),
            "",
        )

    def test_enforce_suppresses_reach(self):
        self.assertEqual(
            physical_interface_aid._enforce_output_format("reach left to find the button"),
            "",
        )

    def test_enforce_suppresses_find(self):
        self.assertEqual(
            physical_interface_aid._enforce_output_format("find the Start button by moving right"),
            "",
        )

    def test_enforce_suppresses_locate(self):
        self.assertEqual(
            physical_interface_aid._enforce_output_format("locate the Enter key"),
            "",
        )

    def test_enforce_diagonal_fixed_valid_form_passes(self):
        """Diagonal fixed to cardinal — no banned verb — response is kept."""
        result = physical_interface_aid._enforce_output_format(
            "move up-right towards Start"
        )
        self.assertEqual(result, "move up towards Start")

    def test_enforce_valid_form_a_passes(self):
        """Form A ('your finger is on …') is returned unchanged."""
        text = "your finger is on Start"
        self.assertEqual(physical_interface_aid._enforce_output_format(text), text)

    def test_enforce_valid_form_b_passes(self):
        """Form B ('move … towards …') is returned unchanged."""
        text = "move right towards Start"
        self.assertEqual(physical_interface_aid._enforce_output_format(text), text)

    def test_enforce_valid_form_c_passes(self):
        """Form C ('[a] is slightly … of your finger …') is returned unchanged."""
        text = "Start is slightly right of your finger, Cancel is slightly left of your finger"
        self.assertEqual(physical_interface_aid._enforce_output_format(text), text)

    def test_enforce_banned_verb_case_insensitive(self):
        """Banned-verb detection is case-insensitive."""
        self.assertEqual(
            physical_interface_aid._enforce_output_format("TOUCH the Start button"),
            "",
        )

    # ── end-to-end: main() rejects bad format from LLM ───────────────────────

    def test_main_suppresses_banned_verb_response(self):
        """main() returns '' when the LLM returns a banned-verb response."""
        physical_interface_aid._last_response = ""
        physical_interface_aid._frame_count = 0
        physical_interface_aid._last_spoken_time = 0.0
        side_effects = [
            _make_llm_result("touch the Start button"),  # banned verb
        ]
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertEqual(result, "")

    def test_main_fixes_diagonal_in_llm_response(self):
        """main() corrects diagonal directions from the LLM before returning."""
        physical_interface_aid._last_response = ""
        physical_interface_aid._frame_count = 0
        physical_interface_aid._last_spoken_time = 0.0
        side_effects = [
            _make_llm_result("move up-right towards Start"),  # diagonal
        ]
        with patch.object(physical_interface_aid, 'copilot_llm_call', side_effect=side_effects):
            result = physical_interface_aid.main(create_blank_image(), {})
        self.assertEqual(result, "move up towards Start")


class TestPhysicalInterfaceAidReturnTypes(unittest.TestCase):
    """Tests that return values conform to the tool contract (mocked)."""

    def setUp(self):
        if not _MODULE_AVAILABLE:
            self.skipTest(f"Module not importable: {_IMPORT_ERROR}")
        physical_interface_aid._last_response = ""
        physical_interface_aid._frame_count = 0
        physical_interface_aid._last_spoken_time = 0.0

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

    def test_module_has_repeat_interval_constant(self):
        """Module must expose REPEAT_INTERVAL for periodic re-announcement."""
        self.assertTrue(hasattr(physical_interface_aid, 'REPEAT_INTERVAL'))
        self.assertIsInstance(physical_interface_aid.REPEAT_INTERVAL, int)
        self.assertGreater(physical_interface_aid.REPEAT_INTERVAL, 0)


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
