"""Standalone CLI for the hoops cooldown observer.

Watches the leveldb save mtime and appends one JSON line per flush to
the observations JSONL. The shared builder + writer live in
`common/hoops_cooldown_observer.py`; the launcher uses the same module
so observations accumulate whenever the launcher is open. This script
is for headless / launcher-less data-collection runs.

Usage:
    uv run python scripts/observe_hoops_cooldown.py
    uv run python scripts/observe_hoops_cooldown.py --once
    uv run python scripts/observe_hoops_cooldown.py --poll-seconds 2

Output (gitignored): minigames/hoops/assets/observations/cooldown.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.hoops_cooldown_observer import (
    OBSERVATIONS_PATH,
    record_observation,
    save_mtime,
)


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
    parser.add_argument("--out", default=str(OBSERVATIONS_PATH),
                        help=f"Output JSONL path (default: {OBSERVATIONS_PATH})")
    parser.add_argument("--once", action="store_true",
                        help="Read current save once and exit (debug)")
    args = parser.parse_args()
    out_path = Path(args.out)

    if args.once:
        row = record_observation(out_path=out_path)
        if row is None:
            print("Couldn't load save.", file=sys.stderr)
            return 1
        print(_format_log_line(row))
        print(json.dumps(row))
        return 0

    print(f"Observing → {out_path}")
    print("Press Ctrl-C to stop.\n")
    last_mtime = 0.0
    try:
        while True:
            mtime = save_mtime()
            if mtime > last_mtime:
                row = record_observation(out_path=out_path, save_mtime_at=mtime)
                if row is None:
                    # Transient leveldb lock — retry next tick without
                    # advancing last_mtime.
                    time.sleep(args.poll_seconds)
                    continue
                print(_format_log_line(row))
                last_mtime = mtime
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
