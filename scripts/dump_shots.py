"""Export an aggregated snapshot of shots.db to a tracked JSON file.

The DB is tracked in git too (since 2026-06-11, for cross-machine
work), but the snapshot remains useful as a human-readable aggregate:
reviewers see what data the predictor was fit on and how make rates
evolve without opening the binary DB.

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

    cols_now = {r[1] for r in conn.execute("PRAGMA table_info(shots)")}

    # Non-gameplay experiment rows (#8) must not dilute the snapshot's
    # make-rate or appear in the per-session/bucket views — the snapshot
    # is a gameplay aggregate. Guarded: older DBs may predate the column.
    _exp = "experiment_label" in cols_now
    exp_where = " WHERE experiment_label IS NULL" if _exp else ""
    exp_and = " AND experiment_label IS NULL" if _exp else ""

    total_shots = conn.execute(f"SELECT COUNT(*) FROM shots{exp_where}").fetchone()[0]
    total_makes = conn.execute(
        f"SELECT COUNT(*) FROM shots WHERE made = 1{exp_and}"
    ).fetchone()[0]
    # Explore shots (free-shot bob-range sampling while the start prompt
    # is up, #37) are deliberately wild — exclude-able counts let
    # reviewers compute an honest make rate that isn't diluted by
    # data-collection shots. NULL target_source (pre-2026-06-09 rows)
    # counts as non-explore.
    explore_expr = (
        "SUM(CASE WHEN target_source = 'explore' THEN 1 ELSE 0 END)"
        if "target_source" in cols_now else "0"
    )
    explore_makes_expr = (
        "SUM(CASE WHEN target_source = 'explore' AND made = 1 THEN 1 ELSE 0 END)"
        if "target_source" in cols_now else "0"
    )
    sessions = []
    for row in conn.execute(
        "SELECT session_started, COUNT(*) AS n, SUM(made) AS makes, "
        f"{explore_expr} AS explore_shots, {explore_makes_expr} AS explore_makes "
        f"FROM shots{exp_where} GROUP BY session_started ORDER BY session_started"
    ):
        sessions.append({
            "started": row["session_started"],
            "shots": row["n"],
            "makes": int(row["makes"] or 0),
            "explore_shots": int(row["explore_shots"] or 0),
            "explore_makes": int(row["explore_makes"] or 0),
        })

    # Defensively check which late-added columns exist — older DBs opened
    # before _migrate ran wouldn't have them.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(shots)")}
    has_perturbation = "perturbation" in cols
    has_lives = "lives_diff" in cols
    has_traj = "ball_apex_y" in cols
    has_window = "window_w" in cols
    has_score_int = "score_after_int" in cols
    has_predicted_offset = "predicted_offset" in cols
    has_code_commit = "code_commit" in cols
    has_predictor_kind = "predictor_kind" in cols
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
    if has_predicted_offset:
        extra_cols += ", predicted_offset"
    if has_code_commit:
        extra_cols += ", code_commit"
    if has_predictor_kind:
        extra_cols += ", predictor_kind"

    makes = []
    for row in conn.execute(
        "SELECT hoop_x, hoop_y, platform_y, \"offset\", target_y, "
        "       clamped, direction, required_direction"
        + extra_cols + " "
        f"FROM shots WHERE made = 1{exp_and} "
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
        if has_predicted_offset:
            rec["predicted_offset"] = row["predicted_offset"]
        if has_code_commit:
            rec["code_commit"] = row["code_commit"]
        if has_predictor_kind:
            rec["predictor_kind"] = row["predictor_kind"]
        makes.append(rec)

    # Per-(hoop_x, hoop_y) bucket aggregate so reviewers can spot stuck
    # positions without iterating raw rows. Buckets are ±5px. last_session
    # is the most recent session that fired into the bucket — it lets a
    # reviewer distinguish an actively-failing zone from an old 0-make
    # bucket that simply hasn't been revisited (the cumulative shots/makes
    # alone can't: a stale bucket and a stuck one look identical). See the
    # #43 review false-alarms.
    #
    # shots/makes blend EVERY predictor the bucket was ever fired with —
    # including the long-since-superseded trajectory_gp/knn/gp and pre-model
    # NULL rows that make at ~14% in the hard mid-band. That makes a bucket
    # look pesky even where the current default (make_prob) does ~64% (the
    # 2026-06-15 (588,340) "26%" false alarm). mp_shots/mp_makes count only
    # make_prob rows, so reviewers can read the CURRENT bot's make rate per
    # bucket instead of the predictor-blended one. (0 when the DB predates
    # the predictor_kind column.)
    pk_sel = ", predictor_kind" if has_predictor_kind else ""
    buckets = {}
    for row in conn.execute(
        f"SELECT hoop_x, hoop_y, made, session_started{pk_sel} FROM shots "
        f"WHERE hoop_x IS NOT NULL{exp_and}"
    ):
        bx = (row["hoop_x"] // 10) * 10
        by = (row["hoop_y"] // 10) * 10
        key = f"{bx},{by}"
        b = buckets.setdefault(key, {"hoop_x_bucket": bx, "hoop_y_bucket": by,
                                     "shots": 0, "makes": 0,
                                     "mp_shots": 0, "mp_makes": 0,
                                     "last_session": None})
        b["shots"] += 1
        if row["made"]:
            b["makes"] += 1
        if has_predictor_kind and row["predictor_kind"] == "make_prob":
            b["mp_shots"] += 1
            if row["made"]:
                b["mp_makes"] += 1
        ss = row["session_started"]
        if ss and (b["last_session"] is None or ss > b["last_session"]):
            b["last_session"] = ss

    bucket_list = sorted(
        buckets.values(),
        key=lambda b: (b["hoop_x_bucket"], b["hoop_y_bucket"]),
    )

    # #38 data-readiness view: velocity-instrumented rows by firing
    # direction x hoop region (region cuts mirror the direction-policy
    # thresholds in minigames/hoops/main.py at time of writing). The
    # make-probability model needs positives in the hard regions under
    # dir=down before a fit is meaningful.
    vy_coverage = []
    if "platform_vy" in cols_now:
        for row in conn.execute(
            "SELECT required_direction, "
            "CASE WHEN hoop_y >= 530 THEN 'low' "
            "     WHEN hoop_x <= 640 THEN 'band' "
            "     ELSE 'mid_far' END AS region, "
            "COUNT(*) AS n, SUM(made) AS makes "
            f"FROM shots WHERE platform_vy IS NOT NULL{exp_and} "
            "GROUP BY required_direction, region"
        ):
            vy_coverage.append({
                "required_direction": row["required_direction"],
                "region": row["region"],
                "shots": row["n"],
                "makes": int(row["makes"] or 0),
            })

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_shots": total_shots,
        "total_makes": total_makes,
        "make_rate": round(total_makes / total_shots, 3) if total_shots else 0,
        "sessions": sessions,
        "buckets": bucket_list,
        "makes": makes,
        "vy_coverage": vy_coverage,
    }

    OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {len(makes)} makes, {len(sessions)} sessions, {len(bucket_list)} buckets to {OUT_PATH}")


if __name__ == "__main__":
    main()
