"""Backfill bust-aware hit/bust labels on darts.db (#40).

The darts hit detector was score-region-diff based: it fired hit=1 on
any score-region change, so busts (which DROP the score and reset the
streak) were logged as hits. The OCR'd score_increment sign disambiguates
them — a decrease is a bust, an increase is a hit. This recomputes hit
and the new bust column for every historical row where the increment is
known, leaving OCR-failed rows (increment NULL) on their diff-based hit.

Run: `uv run python scripts/backfill_darts_bust.py`
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from minigames.darts.shot_log import open_db, classify_throw_outcome

DB_PATH = ROOT / "minigames" / "darts" / "assets" / "darts.db"


def backfill(conn: sqlite3.Connection) -> tuple[int, int]:
    """Recompute (hit, bust) from score_increment for rows with a known
    increment. Returns (rows_updated, busts_found). diff_changed isn't
    stored per row, but for rows with a non-NULL increment the sign is
    authoritative, so the diff fallback is never needed here."""
    rows = conn.execute(
        "SELECT id, hit, bust, score_increment FROM throws "
        "WHERE score_increment IS NOT NULL"
    ).fetchall()
    updated = 0
    busts = 0
    for rid, hit, bust, increment in rows:
        new_hit, new_bust = classify_throw_outcome(increment, None)
        if new_bust:
            busts += 1
        if new_hit != hit or new_bust != bust:
            conn.execute(
                "UPDATE throws SET hit = ?, bust = ? WHERE id = ?",
                (new_hit, new_bust, rid),
            )
            updated += 1
    conn.commit()
    return updated, busts


def main() -> None:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = open_db(DB_PATH)  # ensures the bust column exists
    updated, busts = backfill(conn)
    print(f"Updated {updated} rows; {busts} busts now labelled (hit=0, bust=1).")
    mislabeled = conn.execute(
        "SELECT COUNT(*) FROM throws WHERE hit=1 AND score_increment < 0"
    ).fetchone()[0]
    print(f"Remaining hit=1 with negative increment (should be 0): {mislabeled}")
    conn.close()


if __name__ == "__main__":
    main()
