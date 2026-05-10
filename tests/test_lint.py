"""Pyflakes correctness checks across the package.

Catches the class of bug exemplified by 996fdcf → 6a8b6b1: a NameError
in `_run_inner` that crashed the bot mid-shot because a variable was
renamed in one place but missed in another. Pyflakes' "undefined
name" check would have flagged it without running the code.

This is NOT a full style linter — we only run pyflakes (correctness
checks like undefined names, unused imports, unused locals), not
opinionated rules. Per CLAUDE.md "no linter or formatter config" —
the spirit of that rule is "don't impose style"; correctness checks
that catch real bugs without execution are a different category.
"""
from __future__ import annotations

import io
from pathlib import Path

import pyflakes.api
import pyflakes.reporter

ROOT = Path(__file__).parent.parent
# Packages to lint. tests/ is excluded — pytest fixtures and the like
# legitimately use names that pyflakes can't trace through @pytest.fixture
# indirection and would noise the signal.
PACKAGES = ["common", "minigames", "ui", "scripts"]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for pkg in PACKAGES:
        files.extend(sorted((ROOT / pkg).rglob("*.py")))
    return files


def test_pyflakes_clean():
    """Run pyflakes against every .py file under the lint-targeted
    packages and fail if any issues are reported. Captures pyflakes'
    own output so the failure message points exactly at the offending
    file:line."""
    out = io.StringIO()
    err = io.StringIO()
    reporter = pyflakes.reporter.Reporter(out, err)

    total_issues = 0
    for path in _python_files():
        # checkPath returns the count of issues found in the file.
        total_issues += pyflakes.api.checkPath(str(path), reporter)

    assert total_issues == 0, (
        f"pyflakes reported {total_issues} issue(s):\n"
        f"{out.getvalue()}{err.getvalue()}"
    )
