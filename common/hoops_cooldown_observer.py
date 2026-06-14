"""Hoops cooldown observation logging — gathers data for issue #22.

Whenever the Idleon save flushes, we want to append one JSON line
capturing OLA[423] (hoops cooldown ticks), OLA[424] (hypothesised
streak counter), and adjacent indices. Across enough save flushes we
should see (streak, cooldown-at-set) pairs that reveal the formula
the game uses to scale cooldown duration with consecutive plays.

Two call sites:
- `scripts/observe_hoops_cooldown.py` — standalone polling CLI
- `ui/launcher/bots_tab.py:refresh_tries` — piggybacks on the launcher's
  existing mtime-tick so observations accumulate whenever the launcher
  is open (bot runs, manual plays, character switches all count).
"""
import glob
import json
import os
import time
from pathlib import Path

from common.idleon_save import SAVE_DIR, load_save


REPO_ROOT = Path(__file__).parent.parent
OBSERVATIONS_PATH = REPO_ROOT / "minigames" / "hoops" / "assets" / "observations" / "cooldown.jsonl"

# OLA indices captured per observation. The keys are the JSONL column
# names; downstream analysis doesn't need a legend.
OLA_INDICES_OF_INTEREST: dict[str, int] = {
    "cooldown_ticks": 423,
    "streak": 424,
    "ola_425": 425,
    "ola_426": 426,
    "ola_434": 434,
    "ola_435": 435,
    "ola_436": 436,
    "darts_cooldown_ticks": 439,
    "darts_plays_today": 440,
    "darts_cooldown_base": 442,
}


def save_mtime(save_dir: str = SAVE_DIR) -> float:
    """Newest mtime among the LevelDB save files, or 0 if unreachable.
    Mirrors the launcher's `_save_mtime` so observations anchor to the
    same notion of save time the launcher uses."""
    try:
        files = (glob.glob(os.path.join(save_dir, "*.log"))
                 + glob.glob(os.path.join(save_dir, "*.ldb")))
        return max((os.path.getmtime(p) for p in files), default=0.0)
    except OSError:
        return 0.0


def _active_char(data: dict) -> str | None:
    """Name of the character with the highest PlayerAwayTime (= most
    recently active). PlayerAwayTime is a timestamp, not a duration —
    same direction as `common.idleon_save.read_minigame_plays_shared`
    (which picks the lowest `now - away`)."""
    best_away: float | None = None
    best_name: str | None = None
    for name, info in (data.get("PlayerDATABASE") or {}).items():
        away = info.get("PlayerAwayTime")
        if not isinstance(away, (int, float)):
            continue
        if best_away is None or away > best_away:
            best_away = float(away)
            best_name = name
    return best_name


def build_observation(data: dict, save_mtime_at: float, now: float) -> dict:
    """Pure builder for one observation row. No IO; easy to test
    against synthetic save dicts."""
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list):
        ola = []
    row: dict = {
        "ts": round(now, 3),
        "save_mtime": round(save_mtime_at, 3) if save_mtime_at else None,
        "save_age": round(now - save_mtime_at, 3) if save_mtime_at else None,
        "active_char": _active_char(data),
    }
    for key, idx in OLA_INDICES_OF_INTEREST.items():
        if 0 <= idx < len(ola) and isinstance(ola[idx], (int, float)):
            row[key] = ola[idx]
        else:
            row[key] = None
    return row


def append_observation(row: dict, out_path: Path = OBSERVATIONS_PATH) -> None:
    """Append one row to the JSONL file (creating parent dirs as needed)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def record_observation(
    out_path: Path = OBSERVATIONS_PATH,
    save_dir: str = SAVE_DIR,
    data: dict | None = None,
    save_mtime_at: float | None = None,
) -> dict | None:
    """Load the save (if `data` isn't passed in), build one observation
    row, append it to `out_path`, and return the row.

    Returns None if the save can't be loaded. Callers that already have
    a decoded `data` dict and the mtime should pass them in to avoid a
    second `load_save()` round-trip.
    """
    if data is None:
        data = load_save(save_dir)
    if data is None:
        return None
    if save_mtime_at is None:
        save_mtime_at = save_mtime(save_dir)
    row = build_observation(data, save_mtime_at, time.time())
    append_observation(row, out_path)
    return row
