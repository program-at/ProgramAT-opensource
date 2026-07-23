"""Focused tests for progressive invocation freshness and replacement."""

import ast
import asyncio
import logging
import threading
import unittest
from pathlib import Path
from typing import Any, Dict


def load_lifecycle_helpers():
    source_path = Path(__file__).resolve().parent / "stream_server.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_progressive_invocation_key",
        "_obsolete_progressive_invocation",
        "_progressive_invocation_is_fresh",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "asyncio": asyncio,
        "Any": Any,
        "Dict": Dict,
        "logger": logging.getLogger("progressive-lifecycle-test"),
        "active_progressive_invocations": {},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


class ProgressiveInvocationLifecycleTests(unittest.TestCase):
    def test_new_invocation_marks_old_results_stale(self):
        namespace = load_lifecycle_helpers()
        key = ("client", "tool")
        old = {
            "invocation_id": "old",
            "cancelled": threading.Event(),
            "task": None,
        }
        namespace["active_progressive_invocations"][key] = old

        namespace["_obsolete_progressive_invocation"]("client", "tool")

        new = {
            "invocation_id": "new",
            "cancelled": threading.Event(),
            "task": None,
        }
        namespace["active_progressive_invocations"][key] = new
        is_fresh = namespace["_progressive_invocation_is_fresh"]

        self.assertTrue(old["cancelled"].is_set())
        self.assertFalse(is_fresh(key, old))
        self.assertTrue(is_fresh(key, new))


if __name__ == "__main__":
    unittest.main()
