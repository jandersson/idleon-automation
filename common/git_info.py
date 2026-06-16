"""Read-only git metadata for bot session logging.

Every bot stamps its DB rows with the code commit that fired the shot, so
queries can correlate behavior changes with outcomes ("did the make rate
move after commit X?"). Extracted from the hoops shot log (#35) because
darts, chopping, and mining imported it from there despite sharing nothing
else with the hoops schema.
"""
import hashlib
import subprocess
from pathlib import Path


def current_code_commit(repo_root: Path) -> str | None:
    """Return the short git commit hash of HEAD, with a dirty-tree marker
    if the working tree has uncommitted changes. None if not in a git repo
    or git isn't available.

    Clean tree -> "<sha>". Dirty tree -> "<sha>-dirty.<fp>", where <fp> is
    an 8-char content hash of the uncommitted changes (the tracked diff vs
    HEAD plus the porcelain change set). The fingerprint exists because the
    base <sha> on a dirty run is just whatever HEAD happened to be — often
    an unrelated auto-commit (a hoops/darts DB refresh), not the code that
    actually ran (#50). The fingerprint makes those rows self-identifying:
    two runs on the same dirty tree share a fingerprint and so group
    together, distinct from any other working-tree state. If the diff hash
    can't be computed it degrades to the plain "<sha>-dirty"."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
        if not status.strip():
            return sha
        fp = _dirty_fingerprint(repo_root, status)
        return f"{sha}-dirty.{fp}" if fp else f"{sha}-dirty"
    except Exception:
        return None


def _dirty_fingerprint(repo_root: Path, status: str) -> str | None:
    """8-char content hash of the uncommitted changes: the tracked diff vs
    HEAD combined with the porcelain status (which carries the set of
    changed and untracked paths). Deterministic for a given working-tree
    state. None if the diff can't be read."""
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return None
    payload = (diff + "\x00" + status).encode("utf-8", "replace")
    return hashlib.sha1(payload).hexdigest()[:8]
