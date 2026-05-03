"""Export an aggregated snapshot of shots.db to a tracked JSON file.

The DB itself is gitignored (binary, grows, churn), but a periodically
committed snapshot lets reviewers — including remote agents that don't
have access to the local DB — see what data the predictor was fit on
and how make rates evolve.

Run: `uv run python scripts/dump_shots.py`
Writes: minigames/hoops/assets/shots_snapshot.json
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "minigames" / "hoops" / "assets" / "shots.db"
OUT_PATH = ROOT / "minigames" / "hoops" / "assets" / "shots_snapshot.json"


def main() -> None:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH} — nothing to dump.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total_shots = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    total_makes = conn.execute("SELECT COUNT(*) FROM shots WHERE made = 1").fetchone()[0]

    sessions = []
    for row in conn.execute(
        "SELECT session_started, COUNT(*) AS n, SUM(made) AS makes "
        "FROM shots GROUP BY session_started ORDER BY session_started"
    ):
        sessions.append({
            "started": row["session_started"],
            "shots": row["n"],
            "makes": int(row["makes"] or 0),
        })

    # Defensively check which late-added columns exist — older DBs opened
    # before _migrate ran wouldn't have them.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(shots)")}
    has_perturbation = "perturbation" in cols
    has_lives = "lives_diff" in cols
    has_traj = "ball_apex_y" in cols
    has_window = "window_w" in cols
    has_score_int = "score_after_int" in cols
    extra_cols = ""
    if has_perturbation:
        extra_cols += ", perturbation"
    if has_lives:
        extra_cols += ", lives_diff"
    if has_traj:
        extra_cols += ", ball_apex_y, ball_x_at_rim_height, ball_landing_x"
    if has_window:
        extra_cols += ", window_w, window_h"
    if has_score_int:
        extra_cols += ", score_before_int, score_after_int, score_increment"

    makes = []
    for row in conn.execute(
        "SELECT hoop_x, hoop_y, platform_y, \"offset\", target_y, "
        "       clamped, direction, required_direction"
        + extra_cols + " "
        "FROM shots WHERE made = 1 "
        "ORDER BY hoop_x, hoop_y, platform_y"
    ):
        rec = {
            "hoop_x": row["hoop_x"],
            "hoop_y": row["hoop_y"],
            "platform_y": row["platform_y"],
            "offset": row["offset"],
            "target_y": row["target_y"],
            "clamped": row["clamped"],
            "direction": row["direction"],
            "required_direction": row["required_direction"],
        }
        if has_perturbation:
            rec["perturbation"] = row["perturbation"]
        if has_lives:
            rec["lives_diff"] = row["lives_diff"]
        if has_traj:
            rec["ball_apex_y"] = row["ball_apex_y"]
            rec["ball_x_at_rim_height"] = row["ball_x_at_rim_height"]
            rec["ball_landing_x"] = row["ball_landing_x"]
        if has_window:
            rec["window_w"] = row["window_w"]
            rec["window_h"] = row["window_h"]
        if has_score_int:
            rec["score_before_int"] = row["score_before_int"]
            rec["score_after_int"] = row["score_after_int"]
            rec["score_increment"] = row["score_increment"]
        makes.append(rec)

    # Per-(hoop_x, hoop_y) bucket aggregate so reviewers can spot stuck
    # positions without iterating raw rows. Buckets are ±5px.
    buckets = {}
    for row in conn.execute(
        "SELECT hoop_x, hoop_y, made FROM shots WHERE hoop_x IS NOT NULL"
    ):
        bx = (row["hoop_x"] // 10) * 10
        by = (row["hoop_y"] // 10) * 10
        key = f"{bx},{by}"
        b = buckets.setdefault(key, {"hoop_x_bucket": bx, "hoop_y_bucket": by, "shots": 0, "makes": 0})
        b["shots"] += 1
        if row["made"]:
            b["makes"] += 1

    bucket_list = sorted(
        buckets.values(),
        key=lambda b: (b["hoop_x_bucket"], b["hoop_y_bucket"]),
    )

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_shots": total_shots,
        "total_makes": total_makes,
        "make_rate": round(total_makes / total_shots, 3) if total_shots else 0,
        "sessions": sessions,
        "buckets": bucket_list,
        "makes": makes,
    }

    OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {len(makes)} makes, {len(sessions)} sessions, {len(bucket_list)} buckets to {OUT_PATH}")


if __name__ == "__main__":
    main()
