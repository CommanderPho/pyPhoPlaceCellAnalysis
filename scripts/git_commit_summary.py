#!/usr/bin/env python3
"""
git_commit_summary.py

A general-purpose git commit summarizing workflow that uses Python's ``ast``
module to extract the names of classes and functions that changed
(added / modified / deleted / moved) between consecutive commits and presents
the results grouped by commit day.

Minor changes such as edits to docstrings, comments, or import/dependency
lines are intentionally ignored so that only structural code changes are
reported.

Usage (standalone):
    python scripts/git_commit_summary.py [--repo REPO_PATH]
                                         [--since YYYY-MM-DD]
                                         [--until YYYY-MM-DD]
                                         [--branch BRANCH]
                                         [--output {text,json}]

Usage (as a library):
    from scripts.git_commit_summary import summarize_commits, format_summary
    daily = summarize_commits(repo_path=".", since="2024-01-01")
    print(format_summary(daily))
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SymbolChange:
    """Represents a meaningful change to a single class or function symbol."""

    name: str
    kind: str        # "class" | "function"
    change_type: str  # "added" | "deleted" | "modified" | "moved"
    file_path: str
    moved_from: Optional[str] = None  # only set when change_type == "moved"

    def __str__(self) -> str:
        if self.change_type == "moved":
            return (
                f"{self.change_type:8s} {self.kind:8s} {self.name!r}  "
                f"({self.moved_from} -> {self.file_path})"
            )
        return f"{self.change_type:8s} {self.kind:8s} {self.name!r}  ({self.file_path})"


@dataclass
class CommitSummary:
    """All symbol-level changes introduced by a single commit."""

    commit_hash: str
    date: date
    author: str
    message: str
    symbol_changes: List[SymbolChange] = field(default_factory=list)

    @property
    def short_hash(self) -> str:
        return self.commit_hash[:8]

    def has_changes(self) -> bool:
        return bool(self.symbol_changes)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _strip_location_info(node: ast.AST) -> None:
    """Remove source-position attributes from every node in the tree in-place."""
    for child in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset", "type_comment"):
            try:
                delattr(child, attr)
            except AttributeError:
                pass


def _strip_docstring(node: ast.AST) -> None:
    """
    Remove the leading docstring (if any) from a function or class body
    in-place so that docstring-only edits are not treated as significant.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        node.body = body[1:]


def _normalize_node(node: ast.AST) -> str:
    """
    Return a canonical string for an AST node with:
      - source positions stripped
      - leading docstrings removed from functions / classes

    Two nodes that differ only in whitespace, comments, or docstrings will
    produce the same canonical string.
    """
    node_copy = copy.deepcopy(node)
    _strip_docstring(node_copy)
    _strip_location_info(node_copy)
    return ast.dump(node_copy, annotate_fields=True)


def _extract_top_level_symbols(
    source: str,
) -> Dict[str, Tuple[str, ast.AST]]:
    """
    Parse *source* and return a mapping::

        qualified_name -> (kind, ast_node)

    where *kind* is ``"class"`` or ``"function"``.

    Only top-level and first-level nested (class-body) symbols are collected.
    Module-level imports and plain expressions are intentionally skipped.

    Returns an empty dict if *source* cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    symbols: Dict[str, Tuple[str, ast.AST]] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols[node.name] = ("function", node)
        elif isinstance(node, ast.ClassDef):
            symbols[node.name] = ("class", node)
            # Include methods / nested classes
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols[f"{node.name}.{child.name}"] = ("function", child)
                elif isinstance(child, ast.ClassDef):
                    symbols[f"{node.name}.{child.name}"] = ("class", child)

    return symbols


def _has_non_trivial_body(node: ast.AST) -> bool:
    """
    Return ``True`` when the body of a function / class contains at least one
    statement that is not a *pass*, a docstring, or an ellipsis literal.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return True

    body = list(node.body)
    # Skip leading docstring
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]

    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Ellipsis)):
            continue
        return True
    return False


def _is_import_only_file(source: str) -> bool:
    """
    Return ``True`` when the file contains *only* import statements
    (plus optional docstrings / comments / assignments to __all__ etc.).
    Such files are treated as dependency / plumbing files and skipped.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring
        if isinstance(node, ast.Assign):
            # Allow simple __all__ = [...] style assignments
            targets = node.targets
            if len(targets) == 1 and isinstance(targets[0], ast.Name):
                if targets[0].id.startswith("_"):
                    continue
        return False
    return True


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(args: Sequence[str], repo_path: str = ".") -> str:
    """Run a git command and return stdout as a string."""
    result = subprocess.run(
        ["git", "-C", repo_path] + list(args),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _get_file_at_commit(commit_hash: str, file_path: str, repo_path: str = ".") -> Optional[str]:
    """
    Return the content of *file_path* at *commit_hash*, or ``None`` if the
    file did not exist at that revision.
    """
    try:
        return _run_git(["show", f"{commit_hash}:{file_path}"], repo_path)
    except subprocess.CalledProcessError:
        return None


def _get_commits(
    repo_path: str = ".",
    since: Optional[str] = None,
    until: Optional[str] = None,
    branch: Optional[str] = None,
) -> List[Dict]:
    """
    Return a list of commit metadata dicts (hash, author, timestamp, message)
    in chronological order (oldest first).
    """
    fmt = "%H%x00%ae%x00%aI%x00%s"
    args = ["log", f"--format={fmt}", "--reverse"]
    if branch:
        args.append(branch)
    if since:
        args += [f"--since={since}"]
    if until:
        args += [f"--until={until}"]

    output = _run_git(args, repo_path)
    commits = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\x00")
        if len(parts) != 4:
            continue
        commit_hash, author, timestamp_str, message = parts
        try:
            ts = datetime.fromisoformat(timestamp_str)
            commit_date = ts.date()
        except ValueError:
            # Skip commits whose timestamp cannot be parsed
            continue
        commits.append(
            {
                "hash": commit_hash,
                "author": author,
                "date": commit_date,
                "message": message,
            }
        )
    return commits


@dataclass
class _ChangedFile:
    """A single Python file touched by a commit."""

    file_path: str
    status: str      # "A" | "D" | "M" | "R"
    old_path: Optional[str] = None  # only set for renames (status == "R")


def _get_changed_python_files(
    commit_hash: str, repo_path: str = "."
) -> List[_ChangedFile]:
    """
    Return a list of :class:`_ChangedFile` objects for all Python files
    touched by *commit_hash* relative to its first parent.

    *status* is one of: ``A`` (added), ``D`` (deleted), ``M`` (modified),
    ``R`` (renamed/moved).
    """
    try:
        output = _run_git(
            ["diff-tree", "--no-commit-id", "-r", "--name-status", "-M", commit_hash],
            repo_path,
        )
    except subprocess.CalledProcessError:
        return []

    results: List[_ChangedFile] = []
    for line in output.strip().splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status_code = parts[0][0]  # first char; rename looks like "R085"
        if status_code == "R" and len(parts) == 3:
            old_path, new_path = parts[1], parts[2]
            if new_path.endswith(".py"):
                results.append(_ChangedFile(file_path=new_path, status="R", old_path=old_path))
            continue
        if len(parts) == 2:
            file_path = parts[1]
            if file_path.endswith(".py"):
                results.append(_ChangedFile(file_path=file_path, status=status_code))
    return results


# ---------------------------------------------------------------------------
# Symbol-change detection
# ---------------------------------------------------------------------------


def _get_symbol_changes_for_file(
    old_source: Optional[str],
    new_source: Optional[str],
    file_path: str,
) -> List[SymbolChange]:
    """
    Compare old and new versions of a Python file and return a list of
    ``SymbolChange`` objects for every *significant* difference.

    Minor changes (docstrings, comments, import lines) are ignored.
    """
    old_symbols = _extract_top_level_symbols(old_source) if old_source else {}
    new_symbols = _extract_top_level_symbols(new_source) if new_source else {}

    changes: List[SymbolChange] = []

    # Deleted symbols
    for name, (kind, node) in old_symbols.items():
        if name not in new_symbols:
            changes.append(SymbolChange(name=name, kind=kind, change_type="deleted", file_path=file_path))

    # Added symbols
    for name, (kind, node) in new_symbols.items():
        if name not in old_symbols:
            if _has_non_trivial_body(node):
                changes.append(SymbolChange(name=name, kind=kind, change_type="added", file_path=file_path))

    # Modified symbols
    for name in old_symbols.keys() & new_symbols.keys():
        old_kind, old_node = old_symbols[name]
        new_kind, new_node = new_symbols[name]
        kind = new_kind
        if _normalize_node(old_node) != _normalize_node(new_node):
            changes.append(SymbolChange(name=name, kind=kind, change_type="modified", file_path=file_path))

    return changes


def _resolve_moves(
    all_changes: List[SymbolChange],
) -> List[SymbolChange]:
    """
    Post-process *all_changes* to detect symbols that were deleted from one
    file and added to another within the same commit — those are reported as
    ``"moved"`` rather than separate add/delete pairs.
    """
    deleted = {c.name: c for c in all_changes if c.change_type == "deleted"}
    added = {c.name: c for c in all_changes if c.change_type == "added"}

    moved_names = set(deleted.keys()) & set(added.keys())

    result: List[SymbolChange] = []
    for change in all_changes:
        if change.name in moved_names:
            if change.change_type == "added":
                result.append(
                    SymbolChange(
                        name=change.name,
                        kind=change.kind,
                        change_type="moved",
                        file_path=change.file_path,
                        moved_from=deleted[change.name].file_path,
                    )
                )
            # Skip the corresponding "deleted" entry
            continue
        result.append(change)
    return result


# ---------------------------------------------------------------------------
# Main summarization workflow
# ---------------------------------------------------------------------------


def summarize_commits(
    repo_path: str = ".",
    since: Optional[str] = None,
    until: Optional[str] = None,
    branch: Optional[str] = None,
) -> Dict[date, List[CommitSummary]]:
    """
    Walk git history in *repo_path* (optionally filtered by *since* / *until*
    date strings in ``YYYY-MM-DD`` format and *branch* name), parse every
    changed Python file with the ``ast`` module, and return a dict mapping
    each commit day to the list of per-commit summaries for that day.

    Only *structural* changes are reported.  Edits that touch only
    docstrings, inline comments, or module-level import statements are
    silently discarded.
    """
    commits = _get_commits(repo_path=repo_path, since=since, until=until, branch=branch)
    daily: Dict[date, List[CommitSummary]] = defaultdict(list)

    for commit_info in commits:
        commit_hash = commit_info["hash"]
        changed_files = _get_changed_python_files(commit_hash, repo_path)

        all_changes: List[SymbolChange] = []
        for entry in changed_files:
            if entry.status == "R":
                # Renamed file — compare old path at parent commit to new path at this commit
                old_source = _get_file_at_commit(f"{commit_hash}^", entry.old_path, repo_path)
                new_source = _get_file_at_commit(commit_hash, entry.file_path, repo_path)
                changes = _get_symbol_changes_for_file(old_source, new_source, entry.file_path)
                all_changes.extend(changes)
            elif entry.status == "A":
                new_source = _get_file_at_commit(commit_hash, entry.file_path, repo_path)
                if new_source and _is_import_only_file(new_source):
                    continue
                changes = _get_symbol_changes_for_file(None, new_source, entry.file_path)
                all_changes.extend(changes)
            elif entry.status == "D":
                old_source = _get_file_at_commit(f"{commit_hash}^", entry.file_path, repo_path)
                changes = _get_symbol_changes_for_file(old_source, None, entry.file_path)
                all_changes.extend(changes)
            else:  # "M" or other
                old_source = _get_file_at_commit(f"{commit_hash}^", entry.file_path, repo_path)
                new_source = _get_file_at_commit(commit_hash, entry.file_path, repo_path)
                if new_source and _is_import_only_file(new_source):
                    continue
                changes = _get_symbol_changes_for_file(old_source, new_source, entry.file_path)
                all_changes.extend(changes)

        resolved = _resolve_moves(all_changes)

        summary = CommitSummary(
            commit_hash=commit_hash,
            date=commit_info["date"],
            author=commit_info["author"],
            message=commit_info["message"],
            symbol_changes=resolved,
        )
        daily[commit_info["date"]].append(summary)

    return dict(daily)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_summary(
    daily_summaries: Dict[date, List[CommitSummary]],
    *,
    show_empty: bool = False,
) -> str:
    """
    Render *daily_summaries* as a human-readable plain-text report.

    Parameters
    ----------
    daily_summaries:
        Mapping returned by :func:`summarize_commits`.
    show_empty:
        When ``True``, commits with no structural changes are still listed
        (with a note that no significant changes were found).
    """
    lines: List[str] = []
    for day in sorted(daily_summaries):
        commits = daily_summaries[day]
        day_has_changes = any(c.has_changes() for c in commits)
        if not day_has_changes and not show_empty:
            continue

        lines.append(f"{'=' * 60}")
        lines.append(f"  {day.isoformat()}  ({len(commits)} commit(s))")
        lines.append(f"{'=' * 60}")

        for commit in commits:
            lines.append(f"  [{commit.short_hash}] {commit.message}  <{commit.author}>")
            if commit.symbol_changes:
                for change in commit.symbol_changes:
                    lines.append(f"      {change}")
            elif show_empty:
                lines.append("      (no significant structural changes)")
            lines.append("")

    return "\n".join(lines)


def format_summary_json(
    daily_summaries: Dict[date, List[CommitSummary]],
) -> str:
    """Serialize *daily_summaries* to a JSON string."""
    output = {}
    for day, commits in sorted(daily_summaries.items()):
        day_key = day.isoformat()
        output[day_key] = []
        for commit in commits:
            output[day_key].append(
                {
                    "hash": commit.commit_hash,
                    "author": commit.author,
                    "message": commit.message,
                    "symbol_changes": [
                        {
                            "name": sc.name,
                            "kind": sc.kind,
                            "change_type": sc.change_type,
                            "file_path": sc.file_path,
                            "moved_from": sc.moved_from,
                        }
                        for sc in commit.symbol_changes
                    ],
                }
            )
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize git commits by listing added/modified/deleted/moved "
            "Python classes and functions, grouped by day."
        )
    )
    parser.add_argument(
        "--repo",
        default=".",
        metavar="REPO_PATH",
        help="Path to the git repository (default: current directory).",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only include commits on or after this date.",
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only include commits on or before this date.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        metavar="BRANCH",
        help="Limit history to a specific branch (default: HEAD).",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        default=False,
        help="Also show commits with no structural changes.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        daily = summarize_commits(
            repo_path=args.repo,
            since=args.since,
            until=args.until,
            branch=args.branch,
        )
    except subprocess.CalledProcessError as exc:
        print(f"git error: {exc.stderr}", file=sys.stderr)
        return 1

    if args.output == "json":
        print(format_summary_json(daily))
    else:
        text = format_summary(daily, show_empty=args.show_empty)
        print(text if text.strip() else "(no structural changes found in the given range)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
