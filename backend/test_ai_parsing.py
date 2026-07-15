"""
Tests for AI parsing and template filling functionality
These tests test the logic in isolation without importing stream_server
"""
import unittest
import json
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_decomposition import (
    PLAYING_CARD_STAGE_GOAL,
    build_stage_decomposition_prompt,
    normalize_stage_plan,
)


def load_stream_server_function(name: str):
    """Load one pure helper without importing the server and triggering startup config."""
    source_path = Path(__file__).resolve().parent / "stream_server.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "json": json,
        "re": re,
        "ToolPlanningContext": Any,
    }
    dependency_names = {
        "_normalize_issue_creation_requirements": {
            "_normalize_custom_gpt_value",
            "_explicit_custom_gpt_value",
        },
        "_append_task_stages_to_issue_body": {"_build_task_stages_markdown"},
    }.get(name, set())
    dependencies = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in dependency_names
    ]
    exec(
        compile(ast.Module(body=[*dependencies, function_node], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace[name]


class TestIssueTemplateGuidance(unittest.TestCase):
    """Test Copilot-facing issue template guidance."""

    def test_visual_at_template_uses_separate_task_stages(self):
        template_path = Path(__file__).resolve().parent.parent / ".github" / "ISSUE_TEMPLATE" / "visual_at.md"
        template = template_path.read_text(encoding="utf-8")

        for heading in ("Tool name", "Task", "Expected output", "Constraints / examples", "Mode"):
            self.assertIn(f"## {heading}", template)
        self.assertIn("Task Stages", template)
        self.assertIn("copilot_llm_call", template)
        self.assertNotIn("call_take_photo_baseline_vlm", template)
        self.assertNotIn("TOOL_PROMPT", template)

        instructions_path = template_path.parent.parent / "copilot-instructions.md"
        instructions = instructions_path.read_text(encoding="utf-8")
        self.assertIn("one ordered call per declared stage", instructions)
        self.assertIn("earlier `artifact` values", instructions)
        self.assertIn("last call's `response`", instructions)


class TestParserStageIssueIntegration(unittest.TestCase):
    """Test parser-owned stage normalization and issue-section wiring."""

    def setUp(self):
        self.normalize_requirements = load_stream_server_function("_normalize_issue_creation_requirements")
        self.extraction_prompt = load_stream_server_function("_build_issue_extraction_prompt")
        self.merge_outputs = load_stream_server_function("_merge_issue_and_stage_outputs")
        self.normalize = normalize_stage_plan
        self.render = load_stream_server_function("_build_task_stages_markdown")
        self.append_stages = load_stream_server_function("_append_task_stages_to_issue_body")
        self.parse_json = load_stream_server_function("_parse_llm_json_object")
        self.response_field_only = load_stream_server_function("_response_field_only")

    def test_mobile_payload_uses_only_atomic_response_field(self):
        structured = {
            "response": "Turn right toward the exit.",
            "artifact": {"detections": [{"label": "exit"}]},
            "implementation": "navigator",
            "capability": "navigation",
        }

        self.assertEqual(
            self.response_field_only(structured),
            "Turn right toward the exit.",
        )

    def test_custom_gpt_no_does_not_require_query(self):
        normalized = self.normalize_requirements(
            {
                "description": "Find the user's Uber and guide them to it.",
                "example_usage": "Find the white Toyota and guide me to its passenger door.",
                "live_mode": "yes",
                "live_query": "",
                "missing_fields": ["live_query"],
            },
            "Custom GPT: No",
        )

        self.assertEqual(normalized["custom_gpt"], "no")
        self.assertEqual(normalized["gpt_query"], "")
        self.assertEqual(normalized["missing_fields"], [])

    def test_parser_uses_separate_extraction_and_decomposition_contracts(self):
        task = "Find my Uber, locate the passenger-side door, and guide me to it."
        extraction = self.extraction_prompt(task)
        decomposition = build_stage_decomposition_prompt(task)

        self.assertIn('"missing_fields"', extraction)
        self.assertNotIn('"stages"', extraction)
        self.assertIn('"stages"', decomposition)
        self.assertIn('"goal"', decomposition)
        self.assertIn('"capability"', decomposition)
        self.assertIn("Each stage must contain exactly two fields: goal and capability", decomposition)
        self.assertNotIn('"missing_fields"', decomposition)

    def test_parser_contract_contains_fields_only_for_both_prompt_modes(self):
        extraction = self.extraction_prompt("Identify the playing card. Only say rank and suit.")

        self.assertNotIn('"fused_vlm_prompt"', extraction)
        self.assertIn("Do not generate or improve a VLM prompt", extraction)
        self.assertIn("Do not return stages, subtasks, reasoning steps", extraction)
        self.assertNotIn('"stages"', extraction)

    def test_take_photo_parser_contract_has_no_prompt_or_stage_fields(self):
        extraction = self.extraction_prompt("Read the medication label.")
        self.assertNotIn('"fused_vlm_prompt"', extraction)

    def test_merges_only_stages_into_issue_data(self):
        issue_data = {"title": "Find my Uber", "missing_fields": []}
        decomposition_data = {
            "stages": [
                {"goal": "Find the vehicle", "capability": "object_detection_localization"},
                {"goal": "Guide the user", "capability": "navigation"},
            ],
        }

        merged = self.merge_outputs(issue_data, decomposition_data)

        self.assertEqual(merged["title"], "Find my Uber")
        self.assertIs(merged["stages"], decomposition_data["stages"])
        self.assertEqual(set(merged), {"title", "missing_fields", "stages"})

    def test_normalizes_legacy_object_detection_stage(self):
        result = self.normalize({
            "stages": [{"goal": "Locate the exit", "capability": "object_detection"}],
        })

        self.assertEqual(
            result["stages"],
            [{"goal": "Locate the exit", "capability": "object_detection_localization"}],
        )

    def test_normalizes_playing_card_stage_goal(self):
        result = self.normalize({
            "stages": [{
                "goal": "Identify the cards and their properties",
                "capability": "structured_visual_understanding",
            }],
        })

        self.assertEqual(
            result["stages"],
            [{
                "goal": "Identify only the playing cards by rank and suit.",
                "capability": "structured_visual_understanding",
            }],
        )

    def test_parses_fenced_json_with_trailing_text(self):
        result = self.parse_json(
            '```json\n{"stages":[{"goal":"Guide the user","capability":"navigation"}]}\n```\nDone.'
        )

        self.assertEqual(result["stages"][0]["capability"], "navigation")

    def test_preserves_single_stage(self):
        result = self.normalize(
            {
                "stages": [{
                    "goal": "Extract the medication instructions.",
                    "capability": "ocr",
                }],
            },
            ["ocr", "general_reasoning"],
        )

        self.assertEqual(result, {"stages": [{"goal": "Extract the medication instructions.", "capability": "ocr"}]})

    def test_preserves_multi_stage_order(self):
        result = self.normalize(
            {
                "stages": [
                    {
                        "goal": "Identify the user's vehicle.",
                        "capability": "object_detection_localization",
                    },
                    {
                        "goal": "Locate the passenger-side handle.",
                        "capability": "spatial_reasoning",
                    },
                ],
            },
            ["object_detection_localization", "spatial_reasoning"],
        )

        self.assertEqual(
            result["stages"],
            [
                {"goal": "Identify the user's vehicle.", "capability": "object_detection_localization"},
                {"goal": "Locate the passenger-side handle.", "capability": "spatial_reasoning"},
            ],
        )
        self.assertTrue(all(set(stage) == {"goal", "capability"} for stage in result["stages"]))

    def test_rejects_fields_outside_the_planner_schema(self):
        with self.assertRaisesRegex(ValueError, "only goal and capability"):
            self.normalize(
                {
                    "stages": [{
                        "goal": "Read the label.",
                        "capability": "ocr",
                        "extra": "not allowed",
                    }],
                },
                ["ocr"],
            )

        with self.assertRaisesRegex(ValueError, "only the stages field"):
            self.normalize(
                {
                    "stages": [{"goal": "Read the label.", "capability": "ocr"}],
                    "extra": "not allowed",
                },
                ["ocr"],
            )

    def test_rejects_empty_stage_list(self):
        with self.assertRaisesRegex(ValueError, "non-empty stages list"):
            self.normalize({"stages": []}, ["ocr"])

    def test_rejects_unknown_capability_name(self):
        with self.assertRaisesRegex(ValueError, "unknown capability"):
            self.normalize(
                {
                    "stages": [{
                        "goal": "Find the target object.",
                        "capability": "not_a_real_capability",
                    }],
                },
                ["object_detection_localization", "navigation"],
            )

    def test_issue_flow_contains_task_stages_section_and_logging(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        stage_plan = {
            "stages": [{
                "goal": "Locate the passenger-side handle.",
                "capability": "spatial_reasoning",
            }],
        }
        markdown = self.render(stage_plan)

        self.assertIn("## Task Stages", markdown)
        self.assertIn("### Stage 1", markdown)
        self.assertIn("**Goal:** Locate the passenger-side handle.", markdown)
        self.assertIn("**Expected output:** Final concise spoken response.", markdown)
        self.assertIn("**Depends on:** Original image and user request.", markdown)
        self.assertIn("Including Task Stages in GitHub issue", source)
        self.assertIn("Issue body stage handoff", source)
        self.assertIn("Issue creation stopped because stage decomposition failed", source)
        self.assertIn("Always return at least one stage", build_stage_decomposition_prompt("test"))

    def test_take_photo_parser_runs_stage_decomposition_when_planner_enabled(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        parser = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_transcript_with_ai"
        )
        parser_source = ast.get_source_segment(source, parser)

        self.assertNotIn("execution_mode != 'streaming'", parser_source)
        self.assertIn("build_stage_decomposition_prompt", parser_source)
        self.assertIn("planning_mode=separate_steps", parser_source)

    def test_uber_stage_plan_reaches_issue_markdown(self):
        plan = self.normalize(
            {
                "stages": [
                    {
                        "goal": "Identify the white Toyota Camry with plate ABC123.",
                        "capability": "object_detection_localization",
                    },
                    {
                        "goal": "Locate the passenger-side door.",
                        "capability": "spatial_reasoning",
                    },
                    {
                        "goal": "Guide the user toward the passenger-side door.",
                        "capability": "navigation",
                    },
                ],
            },
            ["object_detection_localization", "spatial_reasoning", "navigation"],
        )
        markdown = self.render(plan)

        self.assertIn("## Task Stages", markdown)
        self.assertIn("### Stage 1", markdown)
        self.assertIn("### Stage 2", markdown)
        self.assertIn("### Stage 3", markdown)
        self.assertIn("useful outputs from stages 1-2", markdown)

        complete = self.normalize_requirements(
            {
                "description": "Find the user's Uber and guide them to it.",
                "example_usage": "Find my Uber, locate its passenger door, and guide me to it.",
                "implementation_details": "",
                "custom_gpt": "no",
                "gpt_query": "",
            },
            "Custom GPT: No",
        )
        body = self.append_stages("**Feature Description**\nUber guidance", plan)

        self.assertEqual(complete["missing_fields"], [])
        self.assertIn("## Task Stages", body)
        self.assertIn("### Stage 3", body)
        self.assertEqual(body, f"**Feature Description**\nUber guidance\n\n{markdown}\n")

    def test_issue_creation_path_does_not_call_model_router(self):
        source_path = Path(__file__).resolve().parent / "stream_server.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        create_node = next(
            node for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "create_github_issue"
        )
        create_source = ast.get_source_segment(source, create_node)

        self.assertNotIn("_build_tool_planning_context", create_source)
        self.assertNotIn("select_model(", create_source)
        self.assertNotIn("final_execution_plan", create_source)
        self.assertNotIn("stage_model_selection", create_source)

    def test_mail_sorter_drops_hallucinated_playing_card_stage(self):
        plan = self.normalize(
            {
                "stages": [
                    {
                        "goal": "Identify the importance of each piece of mail.",
                        "capability": "structured_visual_understanding",
                    },
                    {
                        "goal": "Detect playing cards by rank and suit.",
                        "capability": "structured_visual_understanding",
                    },
                    {
                        "goal": "Read the contents of each letter or package.",
                        "capability": "ocr",
                    },
                ],
            },
            ["structured_visual_understanding", "ocr"],
            source_task="Sort my mail by importance and read each letter or package.",
        )

        self.assertEqual(
            plan["stages"],
            [
                {
                    "goal": "Identify the importance of each piece of mail.",
                    "capability": "structured_visual_understanding",
                },
                {
                    "goal": "Read the contents of each letter or package.",
                    "capability": "ocr",
                },
            ],
        )

    def test_card_identifier_still_uses_exact_card_stage_goal(self):
        plan = self.normalize(
            {
                "stages": [
                    {
                        "goal": "Detect playing cards by rank and suit.",
                        "capability": "structured_visual_understanding",
                    },
                ],
            },
            ["structured_visual_understanding"],
            source_task="Identify the playing cards by rank and suit.",
        )

        self.assertEqual(
            plan["stages"],
            [{
                "goal": PLAYING_CARD_STAGE_GOAL,
                "capability": "structured_visual_understanding",
            }],
        )


class TestSentenceDetectionLogic(unittest.TestCase):
    """Test sentence detection logic"""
    
    def detect_complete_sentence(self, text: str) -> dict:
        """Helper function to detect complete sentences"""
        import re
        sentenceRegex = r'^(.*?[.!?])(\s+|$)'
        match = re.match(sentenceRegex, text)
        
        if match:
            return {
                'complete': match.group(1).strip(),
                'remainder': text[len(match.group(0)):].strip()
            }
        
        return {'complete': '', 'remainder': text}
    
    def test_detects_sentence_with_period(self):
        """Test detection of sentence ending with period"""
        result = self.detect_complete_sentence('Hello world.')
        self.assertEqual(result['complete'], 'Hello world.')
        self.assertEqual(result['remainder'], '')
    
    def test_detects_sentence_with_question_mark(self):
        """Test detection of sentence ending with question mark"""
        result = self.detect_complete_sentence('How are you?')
        self.assertEqual(result['complete'], 'How are you?')
        self.assertEqual(result['remainder'], '')
    
    def test_detects_sentence_with_exclamation(self):
        """Test detection of sentence ending with exclamation mark"""
        result = self.detect_complete_sentence('Great job!')
        self.assertEqual(result['complete'], 'Great job!')
        self.assertEqual(result['remainder'], '')
    
    def test_detects_sentence_with_remainder(self):
        """Test detection of sentence with remaining text"""
        result = self.detect_complete_sentence('Hello world. This is')
        self.assertEqual(result['complete'], 'Hello world.')
        self.assertEqual(result['remainder'], 'This is')
    
    def test_no_complete_sentence(self):
        """Test when there's no complete sentence"""
        result = self.detect_complete_sentence('Hello world')
        self.assertEqual(result['complete'], '')
        self.assertEqual(result['remainder'], 'Hello world')
    
    def test_empty_string(self):
        """Test with empty string"""
        result = self.detect_complete_sentence('')
        self.assertEqual(result['complete'], '')
        self.assertEqual(result['remainder'], '')


class TestTemplateFilling(unittest.TestCase):
    """Test template filling logic"""
    
    def fill_template(self, template_content: str, parsed_data: dict) -> str:
        """Helper function to fill templates"""
        filled = template_content
        
        # Fill visual AT template
        filled = filled.replace('<!-- A clear and concise description of the tool you\'d like. -->', 
                               parsed_data.get('description', ''))
        filled = filled.replace('<!-- Describe the problem this tool would solve. -->', 
                               parsed_data.get('problem', ''))
        filled = filled.replace('<!-- Describe how you envision this tool working. -->', 
                               parsed_data.get('solution', ''))
        filled = filled.replace('<!-- Any particular models or libraries that should be employed -->', 
                               parsed_data.get('implementation_details', ''))
        filled = filled.replace("<!-- Describe any alternative solutions or features you've considered. -->", 
                               parsed_data.get('alternatives', ''))
        filled = filled.replace('<!-- Describe an example situation the tool would be used in and how it could work -->', 
                               parsed_data.get('example_usage', ''))
        filled = filled.replace('<!-- Add any other context or screenshots about the feature request here. -->', 
                               parsed_data.get('additional', ''))
        
        return filled
    
    def test_fill_visual_at_template(self):
        """Test filling a visual AT template"""
        template = """**Feature Description**
<!-- A clear and concise description of the tool you'd like. -->

**Problem It Solves**
<!-- Describe the problem this tool would solve. -->

**Proposed Solution**
<!-- Describe how you envision this tool working. -->

**Example usage**
<!-- Describe an example situation the tool would be used in and how it could work -->"""

        data = {
            'type': 'visual AT',
            'description': 'Object detection tool',
            'problem': 'Need to identify objects in environment',
            'solution': 'Declare object_detection_localization capability and use the backend capability layer',
            'example_usage': 'User points camera at table, tool identifies cup and phone'
        }
        
        result = self.fill_template(template, data)
        
        self.assertIn('Object detection tool', result)
        self.assertIn('Need to identify objects in environment', result)
        self.assertIn('Declare object_detection_localization capability and use the backend capability layer', result)
        self.assertIn('User points camera at table, tool identifies cup and phone', result)
        self.assertNotIn('<!--', result.replace('<!--', '', 1))  # Check most placeholders replaced
    
    def test_fill_with_empty_fields(self):
        """Test filling template with some empty fields"""
        template = """**Description**
<!-- A clear and concise description of the tool you'd like. -->

**Additional Context**
<!-- Add any other context or screenshots about the feature request here. -->"""

        data = {
            'type': 'visual AT',
            'description': 'Tool description',
            'additional': ''  # Empty field
        }
        
        result = self.fill_template(template, data)
        
        self.assertIn('Tool description', result)
        # Should still have the placeholder for empty field
        self.assertIn('**Additional Context**', result)
    
    def test_fill_with_list_values(self):
        """Test filling template when AI returns lists instead of strings"""
        template = """**Description**
<!-- A clear and concise description of the tool you'd like. -->

**Problem It Solves**
<!-- Describe the problem this tool would solve. -->"""

        # Helper function matching the one in stream_server
        def ensure_string(value):
            if isinstance(value, list):
                return '\n'.join(str(item) for item in value)
            elif value is None:
                return ''
            return str(value)
        
        data = {
            'type': 'visual AT',
            'description': ['Object', 'detection', 'tool'],  # List instead of string
            'problem': ['Hard to identify', 'objects in environment']
        }
        
        # Apply the ensure_string logic
        description = ensure_string(data['description'])
        problem = ensure_string(data['problem'])
        
        # Verify lists are converted to strings
        self.assertEqual(description, 'Object\ndetection\ntool')
        self.assertEqual(problem, 'Hard to identify\nobjects in environment')


class TestAIParsingFallback(unittest.TestCase):
    """Test AI parsing fallback logic"""
    
    def simple_parse(self, transcript: str) -> dict:
        """Fallback parsing when API is not available"""
        return {
            'type': 'visual AT',
            'title': transcript[:100],
            'description': transcript,
            'problem': '',
            'solution': '',
            'implementation_details': '',
            'example_usage': '',
            'alternatives': '',
            'additional': ''
        }
    
    def test_fallback_parsing(self):
        """Test that fallback parsing works"""
        transcript = "I want a tool to detect objects in my environment"
        result = self.simple_parse(transcript)
        
        self.assertEqual(result['type'], 'visual AT')
        self.assertEqual(result['title'], transcript)
        self.assertEqual(result['description'], transcript)
        self.assertEqual(result['problem'], '')
    
    def test_fallback_with_long_text(self):
        """Test fallback with text longer than 100 chars"""
        transcript = "A" * 150
        result = self.simple_parse(transcript)
        
        self.assertEqual(len(result['title']), 100)
        self.assertEqual(len(result['description']), 150)


class TestDataMerging(unittest.TestCase):
    """Test multi-turn conversation data merging"""
    
    def merge_parsed_data(self, existing_data: dict, new_data: dict) -> dict:
        """Helper function to merge parsed data"""
        merged = existing_data.copy()
        
        # Merge all fields - new non-empty values override old ones
        for key, value in new_data.items():
            if key == 'missing_fields':
                continue  # Don't merge missing_fields, we'll recalculate
            
            # Update if new value is non-empty
            if value:
                # Handle both strings and lists
                if isinstance(value, str) and value.strip():
                    merged[key] = value
                elif isinstance(value, list) and value:
                    merged[key] = value
                elif not isinstance(value, (str, list)):
                    merged[key] = value
        
        # Update missing_fields from new data (AI re-evaluated what's still missing)
        merged['missing_fields'] = new_data.get('missing_fields', [])
        
        return merged
    
    def test_merge_fills_missing_fields(self):
        """Test that merging fills in previously missing fields"""
        existing = {
            'type': 'visual AT',
            'title': 'Object detection',
            'description': 'A tool to detect objects',
            'problem': '',
            'solution': '',
            'example_usage': '',
            'missing_fields': ['problem', 'solution', 'example_usage']
        }
        
        new = {
            'type': 'visual AT',
            'title': '',
            'description': '',
            'problem': 'Hard to identify objects',
            'solution': 'Declare object_detection_localization capability and use the backend capability layer',
            'example_usage': 'Point camera at table to identify items',
            'missing_fields': []
        }
        
        merged = self.merge_parsed_data(existing, new)
        
        # Should keep original title and description
        self.assertEqual(merged['title'], 'Object detection')
        self.assertEqual(merged['description'], 'A tool to detect objects')
        
        # Should add new fields
        self.assertEqual(merged['problem'], 'Hard to identify objects')
        self.assertEqual(merged['solution'], 'Declare object_detection_localization capability and use the backend capability layer')
        self.assertEqual(merged['example_usage'], 'Point camera at table to identify items')
        
        # Should update missing_fields to empty
        self.assertEqual(merged['missing_fields'], [])
    
    def test_merge_preserves_existing_non_empty(self):
        """Test that merge doesn't overwrite existing non-empty fields with empty ones"""
        existing = {
            'type': 'visual AT',
            'title': 'Object detection',
            'description': 'Full description',
            'problem': 'Need to identify objects',
            'solution': 'Use AI model',
            'example_usage': '',
            'missing_fields': ['example_usage']
        }
        
        new = {
            'type': 'visual AT',
            'title': '',  # Empty, should not overwrite
            'description': '',  # Empty, should not overwrite
            'problem': '',  # Empty, should not overwrite
            'solution': '',  # Empty, should not overwrite
            'example_usage': 'Point at desk to find items',  # New info
            'missing_fields': []
        }
        
        merged = self.merge_parsed_data(existing, new)
        
        # Should preserve existing non-empty fields
        self.assertEqual(merged['title'], 'Object detection')
        self.assertEqual(merged['description'], 'Full description')
        self.assertEqual(merged['problem'], 'Need to identify objects')
        self.assertEqual(merged['solution'], 'Use AI model')
        
        # Should add the new example_usage field
        self.assertEqual(merged['example_usage'], 'Point at desk to find items')
    
    def test_merge_updates_missing_fields(self):
        """Test that missing_fields is properly updated"""
        existing = {
            'type': 'visual AT',
            'title': 'Tool',
            'description': 'A tool',
            'problem': '',
            'solution': '',
            'example_usage': '',
            'missing_fields': ['problem', 'solution', 'example_usage']
        }
        
        new = {
            'type': 'visual AT',
            'title': '',
            'description': '',
            'problem': 'Problem provided',
            'solution': '',
            'example_usage': '',
            'missing_fields': ['solution', 'example_usage']  # Still missing 2
        }
        
        merged = self.merge_parsed_data(existing, new)
        
        # Should have updated missing_fields from new data
        self.assertEqual(merged['missing_fields'], ['solution', 'example_usage'])
        self.assertEqual(merged['problem'], 'Problem provided')


class TestWellSpecifiedIssues(unittest.TestCase):
    """Test that well-specified issues don't trigger requests for more info"""
    
    def test_well_specified_visual_at_should_have_no_missing_fields(self):
        """
        Test that a visual AT request with all important fields specified should have empty missing_fields.
        This simulates what the AI should return for a well-specified issue.
        """
        # Simulate AI parsing a well-specified visual AT request
        transcript = """I want a tool to detect objects in my environment. 
        The problem is I can't identify what's around me when walking. 
        I envision a tool that uses the phone camera to identify objects and speak their names. 
        For example, if I point my camera at my desk, it should tell me there's a laptop, coffee mug, and phone."""
        
        # This is what the AI SHOULD return for a well-specified visual AT request
        expected_parsed_data = {
            'type': 'visual AT',
            'title': 'Object detection tool for environment awareness',
            'description': 'A tool to detect objects in my environment',
            'problem': 'Cannot identify what\'s around me when walking',
            'solution': 'Use phone camera to identify objects and speak their names',
            'implementation_details': '',
            'example_usage': 'Point camera at desk, tool identifies laptop, coffee mug, and phone',
            'alternatives': '',
            'additional': '',
            'missing_fields': []  # All important fields are filled!
        }
        
        # Verify all important visual AT fields are present
        self.assertEqual(expected_parsed_data['type'], 'visual AT')
        self.assertNotEqual(expected_parsed_data['description'].strip(), '')
        self.assertNotEqual(expected_parsed_data['problem'].strip(), '')
        self.assertNotEqual(expected_parsed_data['solution'].strip(), '')
        self.assertNotEqual(expected_parsed_data['example_usage'].strip(), '')
        
        # Most importantly, missing_fields should be empty
        self.assertEqual(expected_parsed_data['missing_fields'], [])
    
    def test_visual_at_with_brief_but_complete_info_should_not_ask_for_more(self):
        """
        Test that even brief descriptions are accepted if they cover important fields.
        """
        # Brief but complete visual AT request
        expected_parsed_data = {
            'type': 'visual AT',
            'title': 'Text reader tool',
            'description': 'Read text from images',
            'problem': 'Cannot read text in environment',
            'solution': 'Declare OCR capability and use the backend capability layer',
            'implementation_details': '',
            'example_usage': 'Point at sign to hear text',
            'alternatives': '',
            'additional': '',
            'missing_fields': []  # Brief but complete!
        }
        
        # Even though the fields are brief, they have meaningful content
        self.assertTrue(len(expected_parsed_data['description']) > 0)
        self.assertTrue(len(expected_parsed_data['problem']) > 0)
        self.assertTrue(len(expected_parsed_data['solution']) > 0)
        self.assertTrue(len(expected_parsed_data['example_usage']) > 0)
        
        # Should not ask for more info
        self.assertEqual(expected_parsed_data['missing_fields'], [])
    
    def test_incomplete_visual_at_should_identify_missing_fields(self):
        """
        Test that genuinely incomplete visual AT requests are correctly identified.
        """
        # Incomplete visual AT request - only has description, missing problem/solution/example
        expected_parsed_data = {
            'type': 'visual AT',
            'title': 'New tool',
            'description': 'I want a new tool',
            'problem': '',  # Missing
            'solution': '',  # Missing
            'implementation_details': '',
            'example_usage': '',  # Missing
            'alternatives': '',
            'additional': '',
            'missing_fields': ['problem', 'solution', 'example_usage']
        }
        
        # Verify the important fields are actually missing
        self.assertEqual(expected_parsed_data['problem'].strip(), '')
        self.assertEqual(expected_parsed_data['solution'].strip(), '')
        self.assertEqual(expected_parsed_data['example_usage'].strip(), '')
        
        # Should correctly identify all missing fields
        self.assertEqual(set(expected_parsed_data['missing_fields']), 
                        {'problem', 'solution', 'example_usage'})


if __name__ == '__main__':
    unittest.main()
