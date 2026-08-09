"""
tests/test_git_commit_summary.py

Unit tests for the Python AST-based git commit summarizer in
``scripts/git_commit_summary.py``.

These tests exercise the pure-Python parsing logic directly, without
requiring a real git repository.
"""

import ast
import sys
import unittest
from pathlib import Path

# Allow importing the script from the sibling ``scripts/`` directory regardless
# of how the test suite is invoked.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from git_commit_summary import (  # noqa: E402
    CommitSummary,
    SymbolChange,
    _extract_top_level_symbols,
    _get_symbol_changes_for_file,
    _has_non_trivial_body,
    _is_import_only_file,
    _normalize_node,
    _resolve_moves,
    format_summary,
    format_summary_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_first(source: str, name: str) -> ast.AST:
    """Return the first top-level AST node whose name matches."""
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if getattr(node, "name", None) == name:
            return node
    raise KeyError(name)


# ---------------------------------------------------------------------------
# _extract_top_level_symbols
# ---------------------------------------------------------------------------

class TestExtractTopLevelSymbols(unittest.TestCase):

    def test_top_level_function(self):
        src = "def foo():\n    return 1\n"
        symbols = _extract_top_level_symbols(src)
        self.assertIn("foo", symbols)
        self.assertEqual(symbols["foo"][0], "function")

    def test_top_level_class(self):
        src = "class Bar:\n    pass\n"
        symbols = _extract_top_level_symbols(src)
        self.assertIn("Bar", symbols)
        self.assertEqual(symbols["Bar"][0], "class")

    def test_class_method_included(self):
        src = "class MyClass:\n    def method(self):\n        pass\n"
        symbols = _extract_top_level_symbols(src)
        self.assertIn("MyClass.method", symbols)
        self.assertEqual(symbols["MyClass.method"][0], "function")

    def test_imports_not_included(self):
        src = "import os\nfrom pathlib import Path\n"
        symbols = _extract_top_level_symbols(src)
        self.assertEqual(symbols, {})

    def test_syntax_error_returns_empty(self):
        symbols = _extract_top_level_symbols("def (\n")
        self.assertEqual(symbols, {})

    def test_async_function(self):
        src = "async def fetch():\n    pass\n"
        symbols = _extract_top_level_symbols(src)
        self.assertIn("fetch", symbols)
        self.assertEqual(symbols["fetch"][0], "function")

    def test_nested_class_in_class(self):
        src = "class Outer:\n    class Inner:\n        pass\n"
        symbols = _extract_top_level_symbols(src)
        self.assertIn("Outer", symbols)
        self.assertIn("Outer.Inner", symbols)


# ---------------------------------------------------------------------------
# _normalize_node
# ---------------------------------------------------------------------------

class TestNormalizeNode(unittest.TestCase):

    def test_docstring_stripped(self):
        src_with = 'def f():\n    """docstring"""\n    return 1\n'
        src_without = "def f():\n    return 1\n"
        node_with = _parse_first(src_with, "f")
        node_without = _parse_first(src_without, "f")
        self.assertEqual(_normalize_node(node_with), _normalize_node(node_without))

    def test_different_bodies_differ(self):
        src_a = "def f():\n    return 1\n"
        src_b = "def f():\n    return 2\n"
        node_a = _parse_first(src_a, "f")
        node_b = _parse_first(src_b, "f")
        self.assertNotEqual(_normalize_node(node_a), _normalize_node(node_b))

    def test_location_info_stripped(self):
        src = "def f():\n    return 1\n"
        node = _parse_first(src, "f")
        normalized = _normalize_node(node)
        self.assertNotIn("lineno", normalized)


# ---------------------------------------------------------------------------
# _has_non_trivial_body
# ---------------------------------------------------------------------------

class TestHasNonTrivialBody(unittest.TestCase):

    def test_pass_only_is_trivial(self):
        node = _parse_first("def f():\n    pass\n", "f")
        self.assertFalse(_has_non_trivial_body(node))

    def test_ellipsis_only_is_trivial(self):
        node = _parse_first("def f():\n    ...\n", "f")
        self.assertFalse(_has_non_trivial_body(node))

    def test_docstring_only_is_trivial(self):
        node = _parse_first('def f():\n    """stub"""\n', "f")
        self.assertFalse(_has_non_trivial_body(node))

    def test_real_body_is_non_trivial(self):
        node = _parse_first("def f():\n    return 42\n", "f")
        self.assertTrue(_has_non_trivial_body(node))


# ---------------------------------------------------------------------------
# _is_import_only_file
# ---------------------------------------------------------------------------

class TestIsImportOnlyFile(unittest.TestCase):

    def test_pure_imports(self):
        src = "import os\nfrom pathlib import Path\n"
        self.assertTrue(_is_import_only_file(src))

    def test_imports_with_function(self):
        src = "import os\n\ndef foo():\n    pass\n"
        self.assertFalse(_is_import_only_file(src))

    def test_empty_file(self):
        self.assertTrue(_is_import_only_file(""))

    def test_module_docstring_with_imports(self):
        src = '"""Module docstring."""\nimport os\n'
        self.assertTrue(_is_import_only_file(src))


# ---------------------------------------------------------------------------
# _get_symbol_changes_for_file
# ---------------------------------------------------------------------------

class TestGetSymbolChangesForFile(unittest.TestCase):

    # -- added -----------------------------------------------------------------

    def test_new_file_with_function(self):
        new_src = "def foo():\n    return 1\n"
        changes = _get_symbol_changes_for_file(None, new_src, "mod.py")
        names = {c.name for c in changes}
        self.assertIn("foo", names)
        for c in changes:
            if c.name == "foo":
                self.assertEqual(c.change_type, "added")

    def test_new_file_stub_function_not_reported(self):
        new_src = "def placeholder():\n    pass\n"
        changes = _get_symbol_changes_for_file(None, new_src, "mod.py")
        self.assertEqual(changes, [])

    # -- deleted ---------------------------------------------------------------

    def test_deleted_file_reports_deleted_symbols(self):
        old_src = "def foo():\n    return 1\n"
        changes = _get_symbol_changes_for_file(old_src, None, "mod.py")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, "deleted")
        self.assertEqual(changes[0].name, "foo")

    # -- modified --------------------------------------------------------------

    def test_modified_function_body(self):
        old_src = "def foo():\n    return 1\n"
        new_src = "def foo():\n    return 2\n"
        changes = _get_symbol_changes_for_file(old_src, new_src, "mod.py")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, "modified")

    def test_docstring_only_change_ignored(self):
        old_src = 'def foo():\n    """old docstring."""\n    return 1\n'
        new_src = 'def foo():\n    """new docstring."""\n    return 1\n'
        changes = _get_symbol_changes_for_file(old_src, new_src, "mod.py")
        self.assertEqual(changes, [])

    def test_import_change_ignored(self):
        old_src = "import os\n\ndef foo():\n    return 1\n"
        new_src = "import os\nimport sys\n\ndef foo():\n    return 1\n"
        changes = _get_symbol_changes_for_file(old_src, new_src, "mod.py")
        self.assertEqual(changes, [])

    def test_class_added(self):
        old_src = "class A:\n    pass\n"
        new_src = "class A:\n    pass\n\nclass B:\n    def method(self):\n        return 1\n"
        changes = _get_symbol_changes_for_file(old_src, new_src, "mod.py")
        names_added = [c.name for c in changes if c.change_type == "added"]
        self.assertIn("B", names_added)
        self.assertIn("B.method", names_added)

    # -- no change -------------------------------------------------------------

    def test_identical_source_produces_no_changes(self):
        src = "def foo():\n    return 1\n\nclass Bar:\n    x = 1\n"
        changes = _get_symbol_changes_for_file(src, src, "mod.py")
        self.assertEqual(changes, [])


# ---------------------------------------------------------------------------
# _resolve_moves
# ---------------------------------------------------------------------------

class TestResolveMoves(unittest.TestCase):

    def test_move_detected(self):
        changes = [
            SymbolChange(name="helper", kind="function", change_type="deleted", file_path="old.py"),
            SymbolChange(name="helper", kind="function", change_type="added", file_path="new.py"),
        ]
        resolved = _resolve_moves(changes)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].change_type, "moved")
        self.assertEqual(resolved[0].file_path, "new.py")
        self.assertEqual(resolved[0].moved_from, "old.py")

    def test_independent_add_delete_not_moved(self):
        changes = [
            SymbolChange(name="foo", kind="function", change_type="deleted", file_path="a.py"),
            SymbolChange(name="bar", kind="function", change_type="added", file_path="b.py"),
        ]
        resolved = _resolve_moves(changes)
        self.assertEqual(len(resolved), 2)
        change_types = {c.change_type for c in resolved}
        self.assertNotIn("moved", change_types)


# ---------------------------------------------------------------------------
# format_summary / format_summary_json
# ---------------------------------------------------------------------------

class TestFormatSummary(unittest.TestCase):

    def _make_daily(self):
        from datetime import date
        from collections import defaultdict

        sc = SymbolChange(name="MyClass", kind="class", change_type="added", file_path="src/mod.py")
        summary = CommitSummary(
            commit_hash="abc1234567890",
            date=date(2025, 6, 15),
            author="dev@example.com",
            message="Add MyClass",
            symbol_changes=[sc],
        )
        return {date(2025, 6, 15): [summary]}

    def test_text_output_contains_date(self):
        text = format_summary(self._make_daily())
        self.assertIn("2025-06-15", text)

    def test_text_output_contains_symbol(self):
        text = format_summary(self._make_daily())
        self.assertIn("MyClass", text)
        self.assertIn("added", text)

    def test_json_output_is_valid(self):
        import json
        payload = json.loads(format_summary_json(self._make_daily()))
        self.assertIn("2025-06-15", payload)
        changes = payload["2025-06-15"][0]["symbol_changes"]
        self.assertEqual(changes[0]["name"], "MyClass")
        self.assertEqual(changes[0]["change_type"], "added")

    def test_show_empty_includes_empty_commits(self):
        from datetime import date

        empty_summary = CommitSummary(
            commit_hash="dead000000",
            date=date(2025, 7, 1),
            author="x@y.com",
            message="chore: update deps",
            symbol_changes=[],
        )
        daily = {date(2025, 7, 1): [empty_summary]}
        text_hide = format_summary(daily, show_empty=False)
        text_show = format_summary(daily, show_empty=True)
        self.assertEqual(text_hide.strip(), "")
        self.assertIn("no significant structural changes", text_show)


if __name__ == "__main__":
    unittest.main()
