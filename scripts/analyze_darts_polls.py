"""Print the labeled poll context around each darts fire.

Joins darts.db's `polls` table (every main-loop poll, dense) with the
`throws` table (one row per fire, with hit/miss outcome) to show, for
each fire, the conf/y trajectory of the polls leading up. Use this to
look for pre-fire signals that discriminate hit from miss — the signal
the lead-up-conf gate tried (and failed) to be.

Usage:
  uv run python scripts/analyze_darts_polls.py                  # latest session
  uv run python scripts/analyze_darts_polls.py --session 2026-05-24T14:00:00
  uv run python scripts/analyze_darts_polls.py --before 15      # show 15 polls before fire
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "minigames" / "darts" / "assets" / "darts.db"


def _arg(name: str, default):
    if name in sys.argv:
        i = sys.argv.index(name) + 1
        if i < len(sys.argv):
            return type(default)(sys.argv[i])
    return default


def main() -> None:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    session = _arg("--session", "")
    if not session:
        row = conn.execute(
            "SELECT MAX(session_started) FROM throws WHERE session_started IS NOT NULL"
        ).fetchone()
        session = row[0] if row and row[0] else None
        if session is None:
            print("No throws in DB.", file=sys.stderr)
            sys.exit(1)
    before_polls = _arg("--before", 12)

    print(f"Session: {session}")
    print(f"Showing {before_polls} polls before each fire")
    print()

    fires = conn.execute(
        """
        SELECT throw_idx, hit, launch_angle_deg, release_pose_y, release_conf
        FROM throws
        WHERE session_started = ?
        ORDER BY throw_idx
        """,
        (session,),
    ).fetchall()
    if not fires:
        print(f"No throws found for session {session}", file=sys.stderr)
        sys.exit(1)

    fire_polls = conn.execute(
        "SELECT t_ms, conf, match_x, match_y FROM polls "
        "WHERE session_started = ? AND threw = 1 ORDER BY t_ms",
        (session,),
    ).fetchall()
    if len(fire_polls) != len(fires):
        print(f"WARN: {len(fires)} throws vs {len(fire_polls)} threw=1 polls — "
              f"can't 1:1 align. Showing what aligns by order.")

    for i, fire in enumerate(fires):
        throw_idx, hit, launch, rel_y, rel_conf = fire
        if i >= len(fire_polls):
            print(f"--- Throw #{throw_idx}: hit={hit} launch={launch} (no poll match) ---")
            continue
        fire_t, fire_conf, fire_x, fire_y = fire_polls[i]
        outcome = "HIT " if hit == 1 else "MISS"
        launch_str = f"{launch:+.1f}°" if launch is not None else "?"
        print(f"--- Throw #{throw_idx}: {outcome} launch={launch_str} fire_t={fire_t}ms ---")

        leading = conn.execute(
            "SELECT t_ms, conf, match_x, match_y FROM polls "
            "WHERE session_started = ? AND t_ms < ? "
            "ORDER BY t_ms DESC LIMIT ?",
            (session, fire_t, before_polls),
        ).fetchall()
        leading.reverse()
        for p in leading:
            dt = p[0] - fire_t  # negative ms before fire
            marker = " <- TRACK" if p[1] >= 0.65 else ""
            print(f"  t{dt:+5d}ms  conf={p[1]:.2f}  (x={p[2]}, y={p[3]}){marker}")
        print(f"  t  +0ms  conf={fire_conf:.2f}  (x={fire_x}, y={fire_y})  <-- FIRE")
        print()

    print("Legend: 'TRACK' = conf >= 0.65 (sub-RELEASE_THRESHOLD but above no-match baseline ~0.61)")


if __name__ == "__main__":
    main()
