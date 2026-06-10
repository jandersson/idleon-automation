"""Export the predictor-playground dataset to docs/playground_data.json.

The GitHub Pages playground (docs/predictor_playground.html) runs the
repo's actual predictors in the browser over the bot's real shot record.
This exporter feeds it:

  makes    — (hoop_x, hoop_y, platform_y) clean makes, the regression
             predictors' training set (same filter as fetch_makes,
             direction "up" — the playground's frame shows up-fires)
  vy_rows  — (hoop_x, hoop_y, platform_y, platform_vy, made) for the
             make-probability view (same rows fetch_vy_labeled returns)

Run after notable data growth: `uv run python scripts/dump_playground_data.py`
(not part of the per-session auto-commit — the playground doesn't need
to be fresher than the prose around it).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from minigames.hoops.shot_log import open_db, fetch_makes, fetch_vy_labeled

DB_PATH = ROOT / "minigames" / "hoops" / "assets" / "shots.db"
OUT_PATH = ROOT / "docs" / "playground_data.json"


def main() -> None:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = open_db(DB_PATH)
    makes = [
        {"hoop_x": int(hx), "hoop_y": int(hy), "platform_y": int(py)}
        for hy, hx, py in fetch_makes(conn, "up")
    ]
    vy_rows = [
        {"hoop_x": int(hx), "hoop_y": int(hy), "platform_y": int(py),
         "vy": round(float(vy), 1), "made": int(made)}
        for hy, hx, py, vy, made in fetch_vy_labeled(conn)
    ]
    conn.close()
    OUT_PATH.write_text(json.dumps({
        "exported": datetime.now().isoformat(timespec="seconds"),
        "frame_w": 960, "frame_h": 572,
        "makes": makes,
        "vy_rows": vy_rows,
    }, indent=1), encoding="utf-8")
    print(f"Wrote {len(makes)} makes + {len(vy_rows)} vy rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
