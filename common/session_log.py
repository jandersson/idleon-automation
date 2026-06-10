"""Tee stdout to a per-session log file so the bot's output is reviewable later
without the user having to copy-paste from the terminal.

Usage:
    with session_log(LOGS_DIR):
        ... run main loop ...
"""
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str):
        for stream in self.streams:
            try:
                stream.write(s)
            except Exception:
                pass
        # Always flush after writing — otherwise the launcher's piped
        # stdout block-buffers and the user sees the log only when the
        # buffer fills (or the process exits). Per-write flush is fine
        # given how rarely we print compared to the bot's main loop.
        self.flush()

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


@contextmanager
def session_log(logs_dir: Path):
    """Open a session_<timestamp>.log under logs_dir; tee stdout AND
    stderr into it.

    stderr matters: an uncaught exception's traceback goes to stderr,
    and before 2026-06-11 the log captured only stdout — a chopping
    session crashed (or failsafe-aborted) leaving a footer but zero
    evidence of why. The exception line is also written explicitly in
    the footer so even a skim of the log tail shows abnormal exits.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"session_{stamp}.log"
    f = open(log_path, "w", encoding="utf-8", buffering=1)
    f.write(f"# session started {datetime.now().isoformat()}\n")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, f)
    sys.stderr = _Tee(original_stderr, f)
    try:
        yield log_path
    except BaseException as e:
        f.write(f"# session ABNORMAL EXIT: {type(e).__name__}: {e}\n")
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        f.write(f"# session ended {datetime.now().isoformat()}\n")
        f.close()


def latest_log(logs_dir: Path) -> Path | None:
    if not logs_dir.exists():
        return None
    logs = sorted(logs_dir.glob("session_*.log"))
    return logs[-1] if logs else None
