"""Tests that ordered temporal input does not imply a card output schema."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_summarizer


class _Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Automatic:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestTemporalOutputSchema(unittest.TestCase):
    def _fake_google_modules(self):
        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.types = types.SimpleNamespace(
            GenerateContentConfig=_Config,
            AutomaticFunctionCallingConfig=_Automatic,
        )
        google.genai = genai
        return patch.dict(sys.modules, {"google": google, "google.genai": genai})

    def test_generic_temporal_generation_has_no_card_schema(self):
        with self._fake_google_modules():
            config = video_summarizer._image_generation_config(None)
        self.assertNotIn("response_json_schema", config.kwargs)
        self.assertNotIn("response_mime_type", config.kwargs)

    def test_played_card_schema_is_enabled_only_when_declared(self):
        with self._fake_google_modules():
            config = video_summarizer._image_generation_config("played_card_event")
        self.assertEqual(config.kwargs["response_mime_type"], "application/json")
        self.assertIs(
            config.kwargs["response_json_schema"],
            video_summarizer.PLAYED_CARD_RESPONSE_SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
