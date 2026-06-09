"""Populate ball_peak_x for historical shots by re-analysing saved flight frames.

For every row with ball_peak_x IS NULL and a shot_dir that still exists on
disk, re-runs track_ball_trajectory and updates the column. Then applies
the back-drift rule directly to historical rows (which already have
clean_make set from the older landing-x-only rule, so _migrate would not
touch them — preserving the "caller-set values stick" contract).

Run: `uv run python scripts/backfill_peak_x.py`
"""
import sqlite3
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.ball_trajectory import track_ball_trajectory
from minigames.hoops.shot_log import open_db, MAX_BACK_DRIFT_PX

DB_PATH = ROOT / "minigames" / "hoops" / "assets" / "shots.db"


def main() -> None:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    # Trigger migration so ball_peak_x exists on legacy DBs.
    open_db(DB_PATH).close()

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT id, shot_dir FROM shots "
        "WHERE ball_peak_x IS NULL AND shot_dir IS NOT NULL"
    ).fetchall()
    print(f"Candidate rows (peak_x NULL, shot_dir set): {len(rows)}")

    updated = 0
    no_dir = 0
    no_frames = 0
    no_ball = 0
    for rid, shot_dir in rows:
        d = Path(shot_dir)
        if not d.exists():
            no_dir += 1
            continue
        frame_paths = sorted(d.glob("flight_*.png"))
        if not frame_paths:
            no_frames += 1
            continue
        frames = [cv2.imread(str(p)) for p in frame_paths]
        frames = [f for f in frames if f is not None]
        positions = track_ball_trajectory(frames)
        if not positions:
            no_ball += 1
            continue
        peak_x = max(p[1] for p in positions)
        conn.execute(
            "UPDATE shots SET ball_peak_x = ? WHERE id = ?",
            (int(peak_x), rid),
        )
        updated += 1
        if updated % 50 == 0:
            print(f"  ... {updated} updated")
            conn.commit()
    conn.commit()
    print(f"Updated: {updated}")
    print(f"Skipped (dir gone): {no_dir}")
    print(f"Skipped (no frames): {no_frames}")
    print(f"Skipped (no ball detected): {no_ball}")
    conn.close()

    # Reopen via open_db so any IS NULL clean_make rows pick up the new
    # rule. Historical rows with clean_make already set are untouched by
    # _migrate (caller-set contract), so reclassify them directly here.
    print()
    print("Reclassifying historical clean_make=1 rows that fail the back-drift gate...")
    conn = open_db(DB_PATH)
    before = conn.execute("SELECT COUNT(*) FROM shots WHERE clean_make = 1").fetchone()[0]
    total_makes = conn.execute("SELECT COUNT(*) FROM shots WHERE made = 1").fetchone()[0]
    conn.execute(
        """
        UPDATE shots
           SET clean_make = 0
         WHERE clean_make = 1
           AND made = 1
           AND ball_peak_x IS NOT NULL
           AND ball_landing_x IS NOT NULL
           AND (ball_peak_x - ball_landing_x) > ?
        """,
        (MAX_BACK_DRIFT_PX,),
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM shots WHERE clean_make = 1").fetchone()[0]
    print(f"Reclassified {before - after} bounce-makes from clean_make=1 -> 0")
    print(f"After backfill: {after}/{total_makes} makes are clean (was {before}/{total_makes})")
    conn.close()


if __name__ == "__main__":
    main()
