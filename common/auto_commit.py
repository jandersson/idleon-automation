"""Auto-commit (and optionally push) a single tracked file from inside a bot.

Used to keep `shots_snapshot.json` (and any future per-minigame data dumps)
fresh in git without the user having to remember to run a script. Best-effort
— failures are logged and swallowed so the bot doesn't error out on a git
hiccup at the end of a session.

Push window honors the global rule: 07:00–23:59 Europe/Stockholm (only the
small hours 00:00–06:59 are off — the rule is "don't appear to work late",
which means avoiding the middle of the night, not the late evening). Only the
specified file is staged, so unrelated working-tree changes are safe (the
user's mid-edit work is never touched).
"""
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PUSH_WINDOW = (7, 24)  # inclusive start, exclusive end (Europe/Stockholm hours): 07:00–23:59


def _within_push_window() -> bool:
    """Returns True iff Europe/Stockholm local hour is in PUSH_WINDOW.

    Falls back to system local time if the IANA zone isn't installed (Windows
    venvs without `tzdata`). The user runs this on their own machine so local
    time is the right approximation."""
    try:
        h = datetime.now(ZoneInfo("Europe/Stockholm")).hour
    except ZoneInfoNotFoundError:
        h = datetime.now().hour
    return PUSH_WINDOW[0] <= h < PUSH_WINDOW[1]


def _git_has_changes(repo_root: Path, path: str) -> bool:
    """Returns True iff `path` differs from the index/HEAD (including
    untracked files, which `git diff HEAD` alone doesn't see)."""
    res = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", path],
        cwd=repo_root,
        capture_output=True,
    )
    if res.returncode != 0:  # non-zero = differences
        return True
    res = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", path],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return bool(res.stdout.strip())


def commit_file_if_changed(
    repo_root: Path,
    relative_path: str | list[str],
    commit_message: str,
    push: bool = True,
) -> None:
    """Stage + commit `relative_path` (one path or a list — only those
    files) if any has changed, then optionally push. Best-effort, never
    raises."""
    paths = [relative_path] if isinstance(relative_path, str) else list(relative_path)
    try:
        changed = [p for p in paths if _git_has_changes(repo_root, p)]
        if not changed:
            return
        subprocess.run(["git", "add", *changed], cwd=repo_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=repo_root, check=True, capture_output=True,
        )
        print(f"  [auto-commit] committed {', '.join(changed)}")
        if push and _within_push_window():
            res = subprocess.run(
                ["git", "push"], cwd=repo_root, capture_output=True, text=True,
            )
            if res.returncode == 0:
                print("  [auto-commit] pushed")
            else:
                print(f"  [auto-commit] push failed (non-fatal): {res.stderr.strip()}")
        elif push:
            print("  [auto-commit] outside push window, will push later")
    except subprocess.CalledProcessError as e:
        print(f"  [auto-commit] git op failed (non-fatal): {e.stderr.decode(errors='replace').strip() if e.stderr else e}")
    except Exception as e:
        print(f"  [auto-commit] unexpected error (non-fatal): {e}")
