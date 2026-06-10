"""#42: replay the darts polls table with bridged-dy logic.

Diagnosis (2026-06-10): arm_centroid_dy_at_fire was NULL on ~1/3 of
aim-gate fires. Every such fire (except session-first-poll ones) was
preceded exactly one poll earlier by a centroid dropout — the motion
mask collapsed below MIN_AREA (px=1..24 vs ~150-300 normal) because a
slow-moving arm produces a small frame diff, and slow descent is what
the dy band selects for. The fire poll itself always had a valid
centroid. centroid_dy_per_poll bridges single-poll dropouts by
dividing displacement over the gap.

This script replays the recorded polls and counts how many dy-NULL
fires the bridge recovers. At fix time: 22 of 24 recovered, 7 in band;
the 2 remainder were the first poll of a session (no previous frame).
"""
import sqlite3
from pathlib import Path

root = Path(__file__).parent.parent

# Frozen copies of the values this replay was run with (2026-06-10).
# The live code has since moved to time-denominated px/s units
# (arm_motion.centroid_vy_px_s); this script replays the HISTORICAL
# px/poll record and keeps the original per-poll bridge math inline.
DY_AIM_LO, DY_AIM_HI = -8, -5  # px/poll at the ~250ms cadence
MAX_DY_GAP_POLLS = 2


def centroid_dy_per_poll(cur_y, prev_y, gap_polls, max_gap=MAX_DY_GAP_POLLS):
    if cur_y is None or prev_y is None or not 1 <= gap_polls <= max_gap:
        return None
    return round((cur_y - prev_y) / gap_polls)


def main() -> None:
    db = sqlite3.connect(root / "minigames/darts/assets/darts.db")
    cur = db.cursor()
    cur.execute(
        "SELECT DISTINCT session_started FROM throws WHERE aim_mode IS NOT NULL"
    )
    sessions = [r[0] for r in cur.fetchall()]

    recovered, still_null, total_null, in_band = 0, 0, 0, 0
    for s in sessions:
        cur.execute(
            "SELECT t_ms, arm_centroid_y, threw FROM polls "
            "WHERE session_started=? ORDER BY t_ms",
            (s,),
        )
        prev_y, gap = None, 1
        for t_ms, cen_y, threw in cur.fetchall():
            dy_old = (
                cen_y - prev_y
                if cen_y is not None and prev_y is not None and gap == 1
                else None
            )
            dy_new = centroid_dy_per_poll(cen_y, prev_y, gap)
            if threw and dy_old is None:
                total_null += 1
                if dy_new is not None:
                    recovered += 1
                    if DY_AIM_LO <= dy_new <= DY_AIM_HI:
                        in_band += 1
                else:
                    still_null += 1
            if cen_y is not None:
                prev_y, gap = cen_y, 1
            else:
                gap += 1

    print(f"dy-NULL fires in aim-gate sessions: {total_null}")
    print(f"  recovered by 1-poll bridge: {recovered} ({in_band} would be in band)")
    print(f"  still NULL (gap>2 or no centroid at fire): {still_null}")


if __name__ == "__main__":
    main()
