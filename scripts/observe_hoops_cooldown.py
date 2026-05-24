"""Watch the Idleon save and log hoops cooldown observations.

Leave this running while playing hoops — each time the game flushes the
save (character switch, periodic auto-save, exit), one JSON line is
appended to the output file capturing the OLA fields relevant to issue
#22 (deriving the hoops cooldown-duration formula from the streak
counter at OLA[424]).

Logged fields per observation:
  ts                       observer wall-clock (seconds since epoch)
  save_mtime               mtime of the newest leveldb file
  save_age                 ts - save_mtime
  cooldown_ticks           OLA[423], hoops cooldown (neg=playable, pos=on cooldown)
  streak                   OLA[424], hypothesized hoops streak counter
  ola_425, ola_426         currently-zero neighbours (might populate later)
  ola_434, ola_435, ola_436 "fast-ticking?" indices flagged in #22
  darts_cooldown_ticks     OLA[439] cross-reference
  darts_plays_today        OLA[440]
  darts_cooldown_base      OLA[442]
  active_char              most-recently-active character (highest PlayerAwayTime)

Procedure for gathering formula data (issue #22):
  1. Start this script.
  2. Wait for OLA[423] to go negative (cooldown fully expired) so the
     streak counter can reset, if it ever does.
  3. Play one hoops session.
  4. Switch character — this flushes the save → poller logs a row.
  5. Repeat (without letting cooldown expire again) so OLA[424] climbs.

The (streak, cooldown_ticks-at-set) pairs across a few play sequences
should reveal the formula — likely linear or geometric in streak.

Output (gitignored): minigames/hoops/assets/observations/cooldown.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.idleon_save import load_save, SAVE_DIR


OUT_PATH = ROOT / "minigames" / "hoops" / "assets" / "observations" / "cooldown.jsonl"

# OLA indices to capture per observation. Keep the keys descriptive so
# downstream analysis doesn't need a legend.
_OLA_INDICES_OF_INTEREST: dict[str, int] = {
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


def _save_mtime(save_dir: str = SAVE_DIR) -> float:
    """Newest mtime among the LevelDB save files, or 0 if unreachable.
    Mirrors the launcher's `_save_mtime` so observations anchor to the
    same notion of "save time" the launcher uses."""
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


def build_observation(data: dict, save_mtime: float, now: float) -> dict:
    """Pure builder for one observation row. Easy to test against
    synthetic save dicts; no IO."""
    ola = data.get("OptionsListAccount")
    if not isinstance(ola, list):
        ola = []
    row: dict = {
        "ts": round(now, 3),
        "save_mtime": round(save_mtime, 3) if save_mtime else None,
        "save_age": round(now - save_mtime, 3) if save_mtime else None,
        "active_char": _active_char(data),
    }
    for key, idx in _OLA_INDICES_OF_INTEREST.items():
        if 0 <= idx < len(ola) and isinstance(ola[idx], (int, float)):
            row[key] = ola[idx]
        else:
            row[key] = None
    return row


def _format_log_line(row: dict) -> str:
    cd = row.get("cooldown_ticks")
    cd_s = f"{cd / 5:.0f}s" if isinstance(cd, (int, float)) else "?"
    age = row.get("save_age")
    age_str = f"{age:.1f}s" if isinstance(age, (int, float)) else "?"
    return (f"[{time.strftime('%H:%M:%S')}] "
            f"OLA[423]={cd} ({cd_s})  "
            f"OLA[424]={row.get('streak')}  "
            f"darts[440]={row.get('darts_plays_today')}  "
            f"active={row.get('active_char')!r}  "
            f"save_age={age_str}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poll-seconds", type=float, default=1.0,
                        help="Seconds between mtime checks (default: 1)")
    parser.add_argument("--out", default=str(OUT_PATH),
                        help=f"Output JSONL path (default: {OUT_PATH})")
    parser.add_argument("--once", action="store_true",
                        help="Read current save once and exit (debug)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.once:
        mtime = _save_mtime()
        data = load_save()
        if data is None:
            print("Couldn't load save.", file=sys.stderr)
            return 1
        row = build_observation(data, mtime, time.time())
        print(_format_log_line(row))
        print(json.dumps(row))
        return 0

    print(f"Observing → {out_path}")
    print("Press Ctrl-C to stop.\n")
    last_mtime = 0.0
    try:
        while True:
            mtime = _save_mtime()
            if mtime > last_mtime:
                data = load_save()
                if data is None:
                    # Transient leveldb lock or missing save — retry next
                    # tick without advancing last_mtime.
                    time.sleep(args.poll_seconds)
                    continue
                row = build_observation(data, mtime, time.time())
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(_format_log_line(row))
                last_mtime = mtime
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
