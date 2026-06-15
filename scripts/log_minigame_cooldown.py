"""Log minigame (hoops/darts) cooldown-vs-plays-today samples to finalise the
cooldown formula (#22).

The cooldown formula is already known in shape — quadratic in plays-today,
capped at ~900 ticks/180s (common.idleon_save.minigame_cooldown_ticks_for_plays:
12*n^2 + 66*n - 26) — but its CONSTANTS are provisional because the original
observations were sampled ~1s after a save flush and treated as lower bounds.

This logger fixes that: it records the save AGE (now - newest leveldb mtime)
alongside the read cooldown ticks, then backs out the decay (5 ticks/sec) to
reconstruct the value AT the flush — so a sample doesn't have to be captured
within 0.5s, it just has to know how stale it is. Flush the save (switch
character) right after a play so the flush ~ the moment the cooldown was set.

To finalise the constants: on a FRESH day (after the daily reset, when
OLA[424] hoops-plays-today is back near 0), run `--watch`, then play hoops
1..7 times, switching character after each play. Each flush appends one row
with hoops_plays and the reconstructed set-ticks. Fit 12*n^2 + b*n + c (or
re-fit the whole polynomial) to the clean (plays, set_ticks) pairs.

Usage:
    uv run python scripts/log_minigame_cooldown.py          # one sample now
    uv run python scripts/log_minigame_cooldown.py --watch  # auto-log per flush
Writes/appends: minigames/hoops/assets/cooldown_samples.csv
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.idleon_save import (  # noqa: E402
    SAVE_DIR, load_save, minigame_cooldown_ticks_for_plays,
    _OLA_HOOPS_COOLDOWN_TICKS, _OLA_HOOPS_PLAYS_TODAY,
    _OLA_DARTS_COOLDOWN_TICKS, _OLA_DARTS_PLAYS_TODAY, _OLA_TICKS_PER_SECOND,
)

OUT_PATH = ROOT / "minigames" / "hoops" / "assets" / "cooldown_samples.csv"
FIELDS = ["ts", "save_age_s", "hoops_plays", "hoops_cd_read_ticks",
          "hoops_cd_set_ticks", "hoops_cd_set_s", "formula_ticks",
          "darts_plays", "darts_cd_read_ticks"]


def reconstruct_set_ticks(read_ticks: float, save_age_s: float,
                          ticks_per_second: int = _OLA_TICKS_PER_SECOND) -> int:
    """Cooldown ticks at the moment the save was flushed, backing out the
    decay since then. The counter ticks down at `ticks_per_second`, so a
    value read `save_age_s` after the flush was that many ticks higher at the
    flush. Only meaningful while a cooldown is actually running (read_ticks
    above ~0); a negative read means 'playable' and there's nothing to set."""
    return round(read_ticks + save_age_s * ticks_per_second)


def _save_age_seconds(save_dir: str = SAVE_DIR) -> float | None:
    """now - the newest mtime among the leveldb files = how long ago Idleon
    last flushed the save. None if the dir is missing/empty."""
    if not os.path.exists(save_dir):
        return None
    mtimes = [os.path.getmtime(os.path.join(save_dir, f))
              for f in os.listdir(save_dir)
              if os.path.isfile(os.path.join(save_dir, f))]
    if not mtimes:
        return None
    return time.time() - max(mtimes)


def sample(save_dir: str = SAVE_DIR) -> dict | None:
    """One reading: hoops/darts plays-today + cooldown ticks + save age +
    the reconstructed set-ticks and the formula's prediction. None if the
    save can't be read."""
    save_age = _save_age_seconds(save_dir)
    data = load_save(save_dir)
    if data is None or save_age is None:
        return None
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list) or len(ola) <= _OLA_DARTS_PLAYS_TODAY:
        return None
    hoops_plays = int(ola[_OLA_HOOPS_PLAYS_TODAY])
    hoops_read = float(ola[_OLA_HOOPS_COOLDOWN_TICKS])
    set_ticks = reconstruct_set_ticks(hoops_read, save_age)
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "save_age_s": round(save_age, 2),
        "hoops_plays": hoops_plays,
        "hoops_cd_read_ticks": round(hoops_read),
        "hoops_cd_set_ticks": set_ticks if hoops_read > 0 else "",
        "hoops_cd_set_s": round(set_ticks / _OLA_TICKS_PER_SECOND, 1) if hoops_read > 0 else "",
        "formula_ticks": minigame_cooldown_ticks_for_plays(hoops_plays),
        "darts_plays": int(ola[_OLA_DARTS_PLAYS_TODAY]),
        "darts_cd_read_ticks": round(float(ola[_OLA_DARTS_COOLDOWN_TICKS])),
    }


def _append(row: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT_PATH.exists()
    with OUT_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _fmt(row: dict) -> str:
    return (f"plays={row['hoops_plays']:>2}  set~{row['hoops_cd_set_ticks'] or '-':>4}t "
            f"(read {row['hoops_cd_read_ticks']}t @ age {row['save_age_s']}s)  "
            f"formula={row['formula_ticks']}t")


def main() -> None:
    ap = argparse.ArgumentParser(description="Log minigame cooldown samples (#22)")
    ap.add_argument("--watch", action="store_true",
                    help="poll the save and log one row on each new flush "
                         "(switch character after a play to trigger a flush)")
    ap.add_argument("--interval", type=float, default=0.2,
                    help="watch poll interval seconds (default 0.2)")
    args = ap.parse_args()

    if not args.watch:
        row = sample()
        if row is None:
            print("Could not read the save (plyvel missing, or no save at "
                  f"{SAVE_DIR}).", file=sys.stderr)
            sys.exit(1)
        _append(row)
        print(_fmt(row), f"-> {OUT_PATH}")
        return

    print(f"Watching {SAVE_DIR} for flushes — play hoops, then switch character "
          f"after each play. Ctrl+C to stop. Logging to {OUT_PATH}")
    last_mtime = None
    while True:
        age = _save_age_seconds()
        # newest mtime = now - age; detect when it advances (a new flush)
        cur_mtime = (time.time() - age) if age is not None else None
        if cur_mtime is not None and cur_mtime != last_mtime:
            last_mtime = cur_mtime
            row = sample()
            if row is not None:
                _append(row)
                print(_fmt(row))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
