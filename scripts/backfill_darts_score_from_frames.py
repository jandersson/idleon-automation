"""Repair darts score fields by re-OCRing the saved settled frames (#40/#41).

The live bot read the post-throw score from a fresh grab that sometimes
caught the score one render-frame before the leading digit landed (13 read
as 3), faking a negative increment. Those were misread as "busts"; in
truth zero real busts exist (across 545 matched throws, no settled frame
shows score decreasing). The per-throw monitor folders saved the settled
pre_throw.png / post_throw.png frames, which read correctly — this
reconstructs score_before/after, increment, hit, bullseye, and stripe_color
from them.

Conservative: a throw is only updated when both its pre and post frames are
readable AND the reconstructed increment is a valid darts outcome
({0,1,2,3,5}); anything else is left untouched. Matches monitor folders to
throws by throw_idx + the meta.txt timestamp falling just after fired_at.

Run: `uv run python scripts/backfill_darts_score_from_frames.py`
"""
import glob
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.regions import get_region
from common.score_template_ocr import (
    extract_digit_components, load_templates, match_digit,
)
from minigames.darts.main import STRIPE_COLOR_BY_INCREMENT

_DARTS = ROOT / "minigames" / "darts"
DB_PATH = _DARTS / "assets" / "darts.db"
MONITOR_DIR = _DARTS / "assets" / "monitor"
VALID_OUTCOMES = {0, 1, 2, 3, 5}

_TEMPLATES = load_templates(_DARTS / "assets" / "digit_templates")


def ocr_score(frame_path: str) -> int | None:
    img = cv2.imread(frame_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    reg = get_region(_DARTS, "score", w, h)
    if reg is None:
        return None
    crop = img[reg["top"]:reg["top"] + reg["height"], reg["left"]:reg["left"] + reg["width"]]
    comps = extract_digit_components(crop)
    digits = [match_digit(p, _TEMPLATES) for p, _ in comps]
    if not digits or any(d is None for d in digits):
        return None
    return int("".join(str(d) for d in digits))


def _index_folders() -> list[tuple[int, datetime, str]]:
    """(throw_idx, meta_timestamp, folder_path) for every monitor folder."""
    out = []
    for f in glob.glob(str(MONITOR_DIR / "throw_*")):
        nm = re.search(r"throw_(\d+)_(\d+)", Path(f).name)
        meta = Path(f) / "meta.txt"
        if not nm or not meta.exists():
            continue
        m = re.search(r"timestamp=(\S+)", meta.read_text())
        if not m:
            continue
        try:
            dt = datetime.fromisoformat(m.group(1))
        except ValueError:
            continue
        out.append((int(nm.group(1)), dt, f))
    return out


def _match_folder(folders, throw_idx, fired_at) -> str | None:
    """Folder whose idx matches and whose meta time falls just after the
    fire (cooldown + save lag), nearest wins."""
    best = None
    for idx, dt, f in folders:
        if idx != throw_idx:
            continue
        delta = (dt - fired_at).total_seconds()
        if -2 <= delta <= 10 and (best is None or abs(delta) < best[0]):
            best = (abs(delta), f)
    return best[1] if best else None


def main() -> None:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    folders = _index_folders()
    print(f"Indexed {len(folders)} monitor folders.")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, throw_idx, fired_at FROM throws WHERE fired_at IS NOT NULL"
    ).fetchall()

    updated = 0
    no_folder = 0
    unreadable = 0
    invalid = 0
    for r in rows:
        try:
            fired_at = datetime.fromisoformat(r["fired_at"])
        except (ValueError, TypeError):
            continue
        folder = _match_folder(folders, r["throw_idx"], fired_at)
        if folder is None:
            no_folder += 1
            continue
        pre = ocr_score(folder + "/pre_throw.png")
        post = ocr_score(folder + "/post_throw.png")
        if pre is None or post is None:
            unreadable += 1
            continue
        inc = post - pre
        if inc not in VALID_OUTCOMES:
            invalid += 1
            continue
        hit = 1 if inc > 0 else 0
        bullseye = 1 if inc == 5 else 0
        stripe_color = STRIPE_COLOR_BY_INCREMENT.get(inc) if inc > 0 else None
        conn.execute(
            "UPDATE throws SET score_before_int=?, score_after_int=?, "
            "score_increment=?, hit=?, bullseye=?, stripe_color=? WHERE id=?",
            (pre, post, inc, hit, bullseye, stripe_color, r["id"]),
        )
        updated += 1
    conn.commit()

    print(f"Updated {updated} throws from settled frames.")
    print(f"Skipped: no folder match={no_folder}, unreadable frame={unreadable}, "
          f"invalid reconstructed increment={invalid}")
    busts = conn.execute(
        "SELECT COUNT(*) FROM throws WHERE hit=1 AND score_increment < 0"
    ).fetchone()[0]
    invalid_incr = conn.execute(
        "SELECT COUNT(*) FROM throws WHERE score_increment IS NOT NULL "
        "AND score_increment NOT IN (0,1,2,3,5)"
    ).fetchone()[0]
    print(f"After: hit=1 with negative increment={busts}; "
          f"rows with invalid increment remaining={invalid_incr}")
    conn.close()


if __name__ == "__main__":
    main()
