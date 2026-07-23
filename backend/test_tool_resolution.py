"""Focused tests for remote-over-local tool execution precedence."""

import ast
import logging
import tempfile
import unittest
from pathlib import Path


def load_resolver():
    source_path = Path(__file__).resolve().parent / "stream_server.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_tool_code_for_execution"
    )
    namespace = {"ast": ast, "Path": Path, "logger": logging.getLogger("tool-resolution-test")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace, namespace["resolve_tool_code_for_execution"]


class ToolResolutionTests(unittest.TestCase):
    def setUp(self):
        self.namespace, self.resolve = load_resolver()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.namespace["TOOLS_DIR"] = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remote_same_named_tool_overrides_local_prompt(self):
        local = 'TOOL_PROMPT = "local prompt"\n'
        remote = 'TOOL_PROMPT = "updated remote prompt"\n'
        (Path(self.temp_dir.name) / "locate_empty_chairs.py").write_text(local, encoding="utf-8")

        with self.assertLogs("tool-resolution-test", level="INFO") as logs:
            resolved = self.resolve(
                "locate_empty_chairs",
                "tools/locate_empty_chairs.py",
                remote,
                "github_pr",
            )

        self.assertEqual(resolved, remote)
        self.assertIn("source=remote", "\n".join(logs.output))
        self.assertNotIn("local prompt", resolved)

    def test_local_is_used_when_remote_is_missing(self):
        local = 'TOOL_PROMPT = "local fallback prompt"\n'
        (Path(self.temp_dir.name) / "locate_empty_chairs.py").write_text(local, encoding="utf-8")

        with self.assertLogs("tool-resolution-test", level="INFO") as logs:
            resolved = self.resolve("locate_empty_chairs", "", "", "local_only")

        self.assertEqual(resolved, local)
        self.assertIn("source=local_fallback", "\n".join(logs.output))

    def test_invalid_remote_python_uses_local_fallback(self):
        local = 'TOOL_PROMPT = "valid local prompt"\n'
        (Path(self.temp_dir.name) / "locate_empty_chairs.py").write_text(local, encoding="utf-8")

        with self.assertLogs("tool-resolution-test", level="INFO") as logs:
            resolved = self.resolve(
                "locate_empty_chairs",
                "tools/locate_empty_chairs.py",
                "def broken(:\n",
            )

        self.assertEqual(resolved, local)
        self.assertIn("source=local_fallback", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
