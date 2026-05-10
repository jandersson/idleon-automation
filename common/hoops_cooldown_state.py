"""Client-side cache of the hoops cooldown set-at timestamp.

Idleon doesn't always flush the save to disk right after a hoops play —
the cooldown is in memory but `OLA[423]` won't reflect it for seconds
or minutes. The launcher would otherwise display "ready" off a stale
save snapshot until the next flush.

When the bot exits a hoops session we know, immediately and exactly,
that a hoops attempt was just consumed and the cooldown was set. We
write a small JSON file with `(set_at, duration_seconds)`; the launcher
extrapolates from that as a fallback when it's more recent than what
the save shows.

State file lives under `data/` (gitignored, per-user state).
"""
import json
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "hoops_cooldown_state.json"

# Full hoops cooldown duration, in seconds. Observed empirically as 1586
# ticks at 5 ticks/sec when the cooldown was just set (see GitHub
# issue #20 for the discovery walkthrough). Talents / items can reduce
# the effective duration in-game; for now this is the post-talent
# observed value for the working save. If your duration differs, the
# save-side reading will eventually correct the display once flushed.
DEFAULT_COOLDOWN_DURATION_SECONDS = 317.2


def write_set_now(
    path: Path = DEFAULT_PATH,
    duration_seconds: float = DEFAULT_COOLDOWN_DURATION_SECONDS,
) -> None:
    """Mark the cooldown as just-set (now) with the given full duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"set_at": time.time(), "duration_seconds": duration_seconds}
    path.write_text(json.dumps(payload), encoding="utf-8")


def read(path: Path = DEFAULT_PATH) -> dict | None:
    """Return {"set_at": float, "duration_seconds": float} or None if no
    state file exists / it's unreadable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("set_at"), (int, float)):
        return None
    if not isinstance(data.get("duration_seconds"), (int, float)):
        return None
    return data


def remaining_seconds(path: Path = DEFAULT_PATH) -> float | None:
    """Convenience: extrapolated remaining cooldown in seconds, or None
    if no state. Negative when expired."""
    state = read(path)
    if state is None:
        return None
    return float(state["duration_seconds"]) - (time.time() - float(state["set_at"]))
