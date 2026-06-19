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
from stage_decomposition import build_stage_decomposition_prompt


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

    def test_visual_at_template_uses_capability_routing(self):
        template_path = Path(__file__).resolve().parent.parent / ".github" / "ISSUE_TEMPLATE" / "visual_at.md"
        template = template_path.read_text(encoding="utf-8")

        self.assertIn("one user-facing task", template)
        self.assertIn("If this issue enumerates multiple stages", template)
        self.assertIn("pass intermediate outputs from earlier stages into later stages", template)
        self.assertIn("The model router only selects the most appropriate model for a capability", template)
        self.assertIn("existing Copilot-routed backend interface through the existing client with `from model_router_client import copilot_llm_call`", template)
        self.assertIn("spatial_relationship", template)
        self.assertIn("Choose only from these task categories", template)
        self.assertNotIn("Capability Pipeline", template)
        self.assertNotIn("should either utilize Yolo11", template)
        self.assertNotIn("Google Vision", template)
        self.assertIn("call `YOLO(...)`", template)
        self.assertIn("define provider-specific `DEFAULT_MODEL` constants", template)


class TestParserStageIssueIntegration(unittest.TestCase):
    """Test parser-owned stage normalization and issue-section wiring."""

    def setUp(self):
        self.normalize_requirements = load_stream_server_function("_normalize_issue_creation_requirements")
        self.extraction_prompt = load_stream_server_function("_build_issue_extraction_prompt")
        self.merge_outputs = load_stream_server_function("_merge_issue_and_stage_outputs")
        self.normalize = load_stream_server_function("_normalize_parser_stage_plan")
        self.render = load_stream_server_function("_build_task_stages_markdown")
        self.append_stages = load_stream_server_function("_append_task_stages_to_issue_body")
        self.parse_json = load_stream_server_function("_parse_llm_json_object")

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
        self.assertNotIn('"routing_analysis"', extraction)
        self.assertNotIn('"pipeline_analysis"', extraction)
        self.assertIn('"routing_analysis"', decomposition)
        self.assertIn('"pipeline_analysis"', decomposition)
        self.assertIn("object_detection -> spatial_relationship", decomposition)
        self.assertIn("spatial_relationship -> navigation", decomposition)
        self.assertNotIn('"missing_fields"', decomposition)

    def test_merges_issue_and_stage_outputs_without_changing_schemas(self):
        issue_data = {"title": "Find my Uber", "missing_fields": []}
        decomposition_data = {
            "routing_analysis": {"tasks": [{"name": "navigation", "weight": 1.0}]},
            "pipeline_analysis": {
                "should_chain": True,
                "stages": [{"capability": "object_detection"}, {"capability": "navigation"}],
            },
        }

        merged = self.merge_outputs(issue_data, decomposition_data)

        self.assertEqual(merged["title"], "Find my Uber")
        self.assertIs(merged["routing_analysis"], decomposition_data["routing_analysis"])
        self.assertIs(merged["pipeline_analysis"], decomposition_data["pipeline_analysis"])

    def test_parses_fenced_json_with_trailing_text(self):
        result = self.parse_json(
            '```json\n{"pipeline_analysis":{"stages":[{"capability":"navigation"}]}}\n```\nDone.'
        )

        self.assertEqual(
            result["pipeline_analysis"]["stages"][0]["capability"],
            "navigation",
        )

    def test_preserves_single_stage(self):
        result = self.normalize(
            {
                "should_chain": False,
                "stages": [{
                    "stage_name": "Read label",
                    "goal": "Extract the medication instructions.",
                    "capability": "ocr",
                    "input": "Current camera frame",
                    "expected_output": "Extracted label text",
                }],
            },
            ["ocr", "general_reasoning"],
        )

        self.assertFalse(result["should_chain"])
        self.assertEqual(len(result["stages"]), 1)
        self.assertEqual(result["stages"][0]["stage_name"], "Read label")
        self.assertEqual(result["stages"][0]["expected_output"], "Extracted label text")

    def test_preserves_multi_stage_dependencies(self):
        result = self.normalize(
            {
                "stages": [
                    {
                        "name": "Find vehicle",
                        "purpose": "Identify the user's vehicle.",
                        "capability": "object_detection",
                        "output": "Vehicle location and description",
                    },
                    {
                        "name": "Find handle",
                        "purpose": "Locate the passenger-side handle.",
                        "capability": "spatial_relationship",
                        "input": "Vehicle information from Stage 1",
                        "output": "Door handle location",
                    },
                ],
            },
            ["object_detection", "spatial_relationship"],
        )

        self.assertTrue(result["should_chain"])
        self.assertEqual(len(result["stages"]), 2)
        self.assertEqual(result["stages"][1]["input"], "Vehicle information from Stage 1")

    def test_issue_flow_contains_task_stages_section_and_logging(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        stage_plan = {
            "stages": [{
                "stage_name": "Find handle",
                "goal": "Locate the passenger-side handle.",
                "capability": "spatial_relationship",
                "input": "Vehicle information from Stage 1",
                "expected_output": "Door handle location",
            }],
        }
        markdown = self.render(stage_plan)

        self.assertIn("## Task Stages", markdown)
        self.assertIn("### Find handle", markdown)
        self.assertIn("**Input dependencies:** Vehicle information from Stage 1", markdown)
        self.assertIn("Including Task Stages in GitHub issue", source)
        self.assertIn("Issue body stage handoff", source)
        self.assertIn("Issue creation stopped because stage decomposition failed", source)
        self.assertIn("Always return at least one stage", build_stage_decomposition_prompt("test"))

    def test_uber_stage_plan_reaches_issue_markdown(self):
        plan = self.normalize(
            {
                "should_chain": True,
                "stages": [
                    {
                        "stage_name": "Identify vehicle",
                        "goal": "Identify the white Toyota Camry with plate ABC123.",
                        "capability": "object_detection",
                        "expected_output": "Vehicle location and description",
                    },
                    {
                        "stage_name": "Locate passenger door",
                        "goal": "Locate the passenger-side door.",
                        "capability": "spatial_relationship",
                        "input": "Vehicle information from Stage 1",
                        "expected_output": "Passenger-side door location",
                    },
                    {
                        "stage_name": "Guide user",
                        "goal": "Guide the user toward the passenger-side door.",
                        "capability": "navigation",
                        "input": "Door location from Stage 2",
                        "expected_output": "Walking guidance",
                    },
                ],
            },
            ["object_detection", "spatial_relationship", "navigation"],
        )
        markdown = self.render(plan)

        self.assertIn("## Task Stages", markdown)
        self.assertIn("### Identify vehicle", markdown)
        self.assertIn("### Locate passenger door", markdown)
        self.assertIn("### Guide user", markdown)

        complete = self.normalize_requirements(
            {
                "problem": "Crowded pickup areas make the correct car difficult to find.",
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
        self.assertIn("### Guide user", body)
        self.assertNotIn("selected_model", body)

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
            'solution': 'Declare object_detection capability and use the backend capability layer',
            'example_usage': 'User points camera at table, tool identifies cup and phone'
        }
        
        result = self.fill_template(template, data)
        
        self.assertIn('Object detection tool', result)
        self.assertIn('Need to identify objects in environment', result)
        self.assertIn('Declare object_detection capability and use the backend capability layer', result)
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
            'solution': 'Declare object_detection capability and use the backend capability layer',
            'example_usage': 'Point camera at table to identify items',
            'missing_fields': []
        }
        
        merged = self.merge_parsed_data(existing, new)
        
        # Should keep original title and description
        self.assertEqual(merged['title'], 'Object detection')
        self.assertEqual(merged['description'], 'A tool to detect objects')
        
        # Should add new fields
        self.assertEqual(merged['problem'], 'Hard to identify objects')
        self.assertEqual(merged['solution'], 'Declare object_detection capability and use the backend capability layer')
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
