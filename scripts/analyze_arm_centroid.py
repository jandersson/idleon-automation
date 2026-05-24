"""Surface the arm-centroid signal at fire time, split by hit/miss.

Joins polls + throws and reports arm_centroid_y at the fire moment for
each labeled fire, grouped by hit-vs-miss (via launch_angle_deg sign:
negative = forward-release hit, positive = top-of-swing apex miss). If
arm_centroid_y discriminates cleanly, the #25 work has its signal.

Also reports the centroid trajectory in the polls leading up to each
fire so we can see velocity / direction signals separate from the
fire-instant value.

Usage:
  uv run python scripts/analyze_arm_centroid.py
  uv run python scripts/analyze_arm_centroid.py --session 2026-05-24T14:55:28
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "minigames" / "darts" / "assets" / "darts.db"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="", help="Limit to one session (default: all sessions with arm-centroid data)")
    parser.add_argument("--lead", type=int, default=5, help="Polls before each fire to print")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))

    # Sessions that have arm-centroid data (i.e. ran under the new instrumentation)
    where_session = "AND session_started = ?" if args.session else ""
    params = (args.session,) if args.session else ()
    sessions = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT session_started FROM polls "
            f"WHERE arm_centroid_y IS NOT NULL {where_session}",
            params,
        )
    ]
    if not sessions:
        print("No sessions with arm-centroid data yet. Run a darts session under "
              "the new build to populate polls.arm_centroid_y / .arm_pixel_count.")
        return

    print(f"Analyzing {len(sessions)} session(s) with arm-centroid data")
    print()

    buckets: dict[str, list[dict]] = defaultdict(list)

    for s in sessions:
        throws = list(
            conn.execute(
                "SELECT fired_at, launch_angle_deg, hit FROM throws "
                "WHERE session_started=? AND launch_angle_deg IS NOT NULL ORDER BY id",
                (s,),
            )
        )
        fires = list(
            conn.execute(
                "SELECT t_ms, conf, match_y, arm_centroid_y, arm_pixel_count "
                "FROM polls WHERE session_started=? AND threw=1 ORDER BY t_ms",
                (s,),
            )
        )
        if len(throws) != len(fires):
            print(f"WARN session {s}: {len(throws)} throws vs {len(fires)} threw=1 polls — "
                  "skipping alignment", file=sys.stderr)
            continue
        for (fired_at, launch, hit), (t_ms, conf, my, acy, acn) in zip(throws, fires):
            label = "HIT" if launch < 0 else "MISS"
            leads = list(
                conn.execute(
                    "SELECT t_ms, conf, arm_centroid_y, arm_pixel_count "
                    "FROM polls WHERE session_started=? AND t_ms < ? "
                    "ORDER BY t_ms DESC LIMIT ?",
                    (s, t_ms, args.lead),
                )
            )
            leads.reverse()
            buckets[label].append({
                "session": s,
                "fire_t": t_ms,
                "launch": launch,
                "fire_arm_y": acy,
                "fire_arm_n": acn,
                "fire_match_y": my,
                "lead": leads,
            })

    for label in ("HIT", "MISS"):
        rows = buckets[label]
        print(f"--- {label} (n={len(rows)}) ---")
        for d in rows:
            lead_arms = [str(l[2]) if l[2] is not None else "-" for l in d["lead"]]
            lead_areas = [str(l[3]) if l[3] is not None else "-" for l in d["lead"]]
            print(f"  fire: arm_y={d['fire_arm_y']} (area={d['fire_arm_n']}) match_y={d['fire_match_y']}  "
                  f"launch={d['launch']:+.1f}")
            print(f"    lead arm_y: {lead_arms}")
            print(f"    lead area:  {lead_areas}")
        print()
        if rows:
            valid = [d["fire_arm_y"] for d in rows if d["fire_arm_y"] is not None]
            if valid:
                lo, hi = min(valid), max(valid)
                mean = sum(valid) / len(valid)
                print(f"  fire arm_y summary: n={len(valid)} (of {len(rows)} fires), "
                      f"min={lo}, mean={mean:.1f}, max={hi}")
            else:
                print("  no fires with arm_centroid_y measurement (motion below MIN_AREA)")
        print()


if __name__ == "__main__":
    main()
