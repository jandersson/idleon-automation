"""Persisted "global tries" counter shared by chopping + catching + mining.

User-managed: the counter is whatever they last typed in via the launcher.
Stored as a tiny JSON so the launcher can read it on every refresh and
write it on every "Set" click.

Lives in <repo>/data/ (gitignored — per-user state).
"""
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "global_tries.json"


def read(path: Path = DEFAULT_PATH) -> int | None:
    """Return the saved tries count, or None if the file doesn't exist or
    contains nothing usable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get("tries_left")
        return int(val) if val is not None else None
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return None


def write(value: int, path: Path = DEFAULT_PATH) -> None:
    """Persist a new tries count. Caller validates the value (non-negative
    int) — we just write what we're given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tries_left": int(value)}, indent=2), encoding="utf-8")
