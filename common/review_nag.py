"""Count sessions since the last code commit by Claude / the user.

Used to surface a "you've played N sessions since the last code review,
maybe ping Claude" nudge at the end of a bot session, without depending
on any external notification system.

Heuristic: the "last review" is the most recent commit that touched
anything OTHER than the auto-committed snapshot file. Auto-commits of
shots_snapshot.json don't count as reviews — they're just data dumps.
"""
import sqlite3
import subprocess
from pathlib import Path

NAG_THRESHOLD = 10  # session count at which to print the nag


def sessions_since_last_code_commit(
    repo_root: Path,
    db_path: Path,
    snapshot_relative: str,
) -> int:
    """Count distinct session_started timestamps in `db_path` that are newer
    than the most recent commit touching anything except `snapshot_relative`.
    Returns 0 on any error (best-effort, never raises)."""
    try:
        # Last commit author-date that touched files OTHER than the snapshot.
        # `:(exclude)` is git's pathspec magic for "everything except this".
        res = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--",
             ".", f":(exclude){snapshot_relative}"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        )
        last_commit_iso = res.stdout.strip()
        if not last_commit_iso:
            return 0
        # `aI` already gives a strict-ISO format that SQLite can compare lexically
        # because session_started is also stored as ISO. Keep the timezone offset
        # off the SQL side: strip everything from the first '+' or 'Z' on, since
        # session_started is stored as naive local-ISO without an offset.
        cutoff = last_commit_iso.split("+")[0].split("Z")[0]
        conn = sqlite3.connect(str(db_path))
        try:
            n = conn.execute(
                "SELECT COUNT(DISTINCT session_started) FROM shots "
                "WHERE session_started > ?",
                (cutoff,),
            ).fetchone()[0]
            return int(n or 0)
        finally:
            conn.close()
    except Exception:
        return 0


def maybe_print_nag(
    repo_root: Path,
    db_path: Path,
    snapshot_relative: str,
    threshold: int = NAG_THRESHOLD,
) -> None:
    """Print a one-line nudge to stdout if we're past the threshold."""
    n = sessions_since_last_code_commit(repo_root, db_path, snapshot_relative)
    if n >= threshold:
        print(f"  [review-nag] {n} sessions since the last code review — "
              f"consider pinging Claude when convenient.")
