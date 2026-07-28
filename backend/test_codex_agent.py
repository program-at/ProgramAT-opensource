import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codex_agent


class TestCodexAgent(unittest.TestCase):
    def test_prompt_uses_repository_instructions_and_preserves_update(self):
        prompt = codex_agent.build_prompt(
            42,
            "Indoor exit finder",
            "## Task\nFind the most likely indoor exit.",
            "Use the stronger model only when uncertain.",
        )

        self.assertIn("Read and follow .github/copilot-instructions.md.", prompt)
        self.assertIn("Find the most likely indoor exit.", prompt)
        self.assertIn("Use the stronger model only when uncertain.", prompt)
        self.assertIn("Do not create branches, commits, pushes", prompt)

    def test_slug_is_safe_and_bounded(self):
        slug = codex_agent._slug("Indoor Exit Finder! " * 10)
        self.assertRegex(slug, r"^[a-z0-9-]+$")
        self.assertLessEqual(len(slug), 40)

    def test_publish_url_uses_configured_issue_repository(self):
        self.assertEqual(
            codex_agent._github_repo_url("yushanwe/ProgramAT-opensource"),
            "https://github.com/yushanwe/ProgramAT-opensource.git",
        )
        with self.assertRaises(ValueError):
            codex_agent._github_repo_url("https://github.com/wrong/repo")

    def test_cancel_terminates_active_process(self):
        process = Mock()
        process.returncode = None
        process.terminate = Mock()
        process.wait = AsyncMock(return_value=0)
        run = codex_agent.CodexRun(
            "codex-issue-7-test",
            7,
            "codex/issue-7-test",
            codex_agent.Path("/tmp/programat-codex-test"),
            process=process,
        )
        codex_agent.active_runs[7] = run
        try:
            self.assertTrue(asyncio.run(codex_agent.cancel_run(7)))
            self.assertTrue(run.cancelled)
            process.terminate.assert_called_once()
        finally:
            codex_agent.active_runs.clear()


if __name__ == "__main__":
    unittest.main()
