"""Backfill experiment_label for the 2026-05-03 click-position rig (#8).

The click_sweep / click_extreme investigation (settled 2026-05-03,
see docs/cleanup_backlog.md) fired against a STATIC test rig: the hoop
held at one fixed position while click coordinates were varied. Real
gameplay never holds the hoop still — it moves/respawns every shot — so
those sessions are identifiable structurally without a label having
been written at the time:

    a 2026-05-03 session that logged click coordinates, fired >=3 shots,
    and kept the hoop at a single (hoop_x, hoop_y)

That isolates exactly the click-position sessions (62 rows across five
sessions, all 0 makes — consistent with deliberate off-aim clicks) and
leaves normal gameplay (hoop moves => >1 distinct hoop position) and
non-click controlled runs (no click data) untouched. The data can't
cleanly split click_sweep from click_extreme 13 months on, and doesn't
need to — both are the same investigation and are excluded identically;
the single 'click_position' label is the honest granularity. (The sweep
session is still recognisable by its multiple distinct click_y values
if anyone wants to drill in.)

These rows already had clean_make=0 so they never trained a predictor;
this label makes the exclusion explicit and survives any future change
to the clean_make rule.

Run: `uv run python scripts/backfill_experiment_label.py`
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from minigames.hoops.shot_log import open_db

DB_PATH = ROOT / "minigames" / "hoops" / "assets" / "shots.db"

CLICK_RIG_LABEL = "click_position"


def label_click_experiments(conn: sqlite3.Connection) -> int:
    """Tag the 2026-05-03 fixed-hoop click-position rig rows. Returns the
    number of rows newly labelled. Idempotent: only touches rows whose
    experiment_label IS NULL."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(shots)")}
    if "click_x" not in cols:
        # click_x/click_y were dropped from the schema after the May-3
        # experiments; only the historical DB that ran them still has the
        # columns. No column => no click-position rows to label.
        return 0
    cur = conn.execute(
        """
        UPDATE shots
           SET experiment_label = ?
         WHERE experiment_label IS NULL
           AND session_started IN (
                 SELECT session_started FROM shots
                  WHERE substr(session_started, 1, 10) = '2026-05-03'
                    AND click_x IS NOT NULL
                  GROUP BY session_started
                 HAVING COUNT(*) >= 3
                    AND COUNT(DISTINCT hoop_x || ',' || hoop_y) = 1
           )
        """,
        (CLICK_RIG_LABEL,),
    )
    conn.commit()
    return cur.rowcount


def main() -> None:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = open_db(DB_PATH)  # ensures experiment_label column exists
    n = label_click_experiments(conn)
    total = conn.execute(
        "SELECT COUNT(*) FROM shots WHERE experiment_label IS NOT NULL"
    ).fetchone()[0]
    print(f"Labelled {n} new rows as {CLICK_RIG_LABEL!r}; "
          f"{total} rows now carry an experiment_label.")
    for r in conn.execute(
        "SELECT session_started, COUNT(*), SUM(made) FROM shots "
        "WHERE experiment_label IS NOT NULL GROUP BY session_started "
        "ORDER BY session_started"
    ):
        print(f"  {r[0]}  n={r[1]}  makes={r[2]}")
    conn.close()


if __name__ == "__main__":
    main()
