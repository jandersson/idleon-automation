"""Read-only git metadata for bot session logging.

Every bot stamps its DB rows with the code commit that fired the shot, so
queries can correlate behavior changes with outcomes ("did the make rate
move after commit X?"). Extracted from the hoops shot log (#35) because
darts, chopping, and mining imported it from there despite sharing nothing
else with the hoops schema.
"""
import subprocess
from pathlib import Path


def current_code_commit(repo_root: Path) -> str | None:
    """Return the short git commit hash of HEAD, with "-dirty" suffix if
    the working tree has uncommitted changes. None if not in a git repo
    or git isn't available."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return None
