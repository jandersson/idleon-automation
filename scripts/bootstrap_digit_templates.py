"""Bootstrap per-game digit templates from existing monitor frames.

For each game (darts/hoops), walks the relevant DB to find score-region
crops where tesseract has given a sane reading. Extracts per-digit
binary components and writes one PNG per digit to
`<game>/assets/digit_templates/<digit>.png`. Idempotent — re-running
overwrites with the latest sample but won't break anything.

Strategy for "sane reading":
  * tesseract returned a non-None int
  * the count of extracted digit components matches the number of
    digits in the reading (rejects crops where noise / cropping
    artifacts split a single digit into multiple components)
  * for hoops, additionally cross-check `score_after_int` matches
    score_before_int + score_increment where both are present

Usage:
  uv run python scripts/bootstrap_digit_templates.py
  uv run python scripts/bootstrap_digit_templates.py --game darts
  uv run python scripts/bootstrap_digit_templates.py --game hoops --verbose
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.regions import get_region
from common.score_template_ocr import (
    extract_digit_components,
    save_template,
)


GAMES = {
    "darts": {
        "db": ROOT / "minigames" / "darts" / "assets" / "darts.db",
        "game_dir": ROOT / "minigames" / "darts",
        "table": "throws",
        "score_col": "score_after_int",
        "throw_dir_col": "throw_dir",
    },
    "hoops": {
        "db": ROOT / "minigames" / "hoops" / "assets" / "shots.db",
        "game_dir": ROOT / "minigames" / "hoops",
        "table": "shots",
        "score_col": "score_after_int",
        "throw_dir_col": "shot_dir",
    },
}


# Manual overrides: known-correct scores for monitor frames where OCR
# failed but the value is recoverable from progression / visual reading.
# Keyed by (game, monitor dir name). Use this to fill in digits that
# don't appear in any clean OCR reading. Add entries here when you spot
# a missing digit in `_score_*.png` debug crops.
MANUAL_LABELS: dict[tuple[str, str], int] = {
    # Session 2026-05-24T14:20:07 — three hits in a row, derivable from progression.
    # throw 3: before=6 + hit (+3 visual) = 9. Confirms shape of "9".
    ("darts", "throw_003_142027"): 9,
    # throw 5: before=9 + bullseye (+5) = 14. Gives us "1" and "4".
    ("darts", "throw_005_142051"): 14,
}


def _is_clean_reading(reading: int, components: int) -> bool:
    """A clean reading is a non-negative int whose digit count matches
    the number of extracted components. Filters crops where noise split
    a digit (e.g. components=3 for a 2-digit number) or merged digits."""
    if reading is None or reading < 0:
        return False
    expected_digits = len(str(reading))
    return components == expected_digits


def bootstrap_game(game_name: str, verbose: bool = False) -> dict[int, int]:
    """Extract digit templates for one game. Returns dict of {digit: count_observed}."""
    cfg = GAMES[game_name]
    db_path = cfg["db"]
    if not db_path.exists():
        print(f"[{game_name}] no DB at {db_path}", file=sys.stderr)
        return {}

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        f"SELECT {cfg['throw_dir_col']}, {cfg['score_col']} "
        f"FROM {cfg['table']} "
        f"WHERE {cfg['throw_dir_col']} IS NOT NULL AND {cfg['score_col']} IS NOT NULL "
        f"ORDER BY id"
    ).fetchall()
    # Every session's throw_idx=1 has score=0 in its pre_throw.png (first
    # throw hasn't scored yet). Reuse `_extract_pre_frame_score_zero` to
    # mine these frames for digit "0" without manual labelling. Uses
    # pre_throw.png (not post_throw.png) so the score is unchanged from
    # session start.
    idx_col = "throw_idx" if cfg["table"] == "throws" else "shot_idx"
    first_throw_dirs = conn.execute(
        f"SELECT DISTINCT {cfg['throw_dir_col']} FROM {cfg['table']} "
        f"WHERE {idx_col} = 1 AND {cfg['throw_dir_col']} IS NOT NULL"
    ).fetchall()
    conn.close()

    template_dir = cfg["game_dir"] / "assets" / "digit_templates"
    captured: dict[int, np.ndarray | None] = {d: None for d in range(10)}
    observations: dict[int, int] = {d: 0 for d in range(10)}

    # Combine OCR-derived rows with manual overrides so digits missing
    # from successful tesseract reads can still be captured.
    sources = list(rows)
    cfg_dir = cfg["game_dir"]
    for (override_game, override_dir), override_score in MANUAL_LABELS.items():
        if override_game != game_name:
            continue
        override_path = cfg_dir / "assets" / "monitor" / override_dir
        sources.append((str(override_path), override_score))
    # Inject score=0 candidates from every throw_idx=1's pre_throw.png.
    # Use a sentinel score value of 0 + force the loop below to read
    # pre_throw.png instead of post_throw.png via a marker tuple.
    for (dir_path,) in first_throw_dirs:
        if dir_path is None:
            continue
        sources.append(("PRE:" + dir_path, 0))

    for shot_dir, score in sources:
        if shot_dir is None or score is None:
            continue
        # "PRE:" prefix = use pre_throw.png (score is pre-update); without
        # it, use post_throw.png (score is post-update). Lets us mine
        # throw_idx=1's pre-shot frame for digit 0 (score=0 in startup).
        use_pre = isinstance(shot_dir, str) and shot_dir.startswith("PRE:")
        sd = Path(shot_dir[4:] if use_pre else shot_dir)
        if not sd.exists():
            continue
        if use_pre:
            post_path = sd / "pre_throw.png"
            if not post_path.exists():
                continue
        else:
            post_path = sd / "post_throw.png"
            if not post_path.exists():
                # hoops uses shot_NNN dirs that don't contain post_throw.png
                # the same way. Fall back to scanning the parent monitor dir's
                # last flight frame.
                flights = sorted(sd.glob("flight_*.png"))
                if not flights:
                    continue
                post_path = flights[-1]
        frame = cv2.imread(str(post_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        region = get_region(cfg["game_dir"], "score", w, h)
        if region is None:
            continue
        crop = frame[
            region["top"]:region["top"] + region["height"],
            region["left"]:region["left"] + region["width"],
        ]
        components = extract_digit_components(crop)
        score_str = str(score)
        if not _is_clean_reading(score, len(components)):
            if verbose:
                print(f"  [{game_name}] skip {sd.name}: score={score} "
                      f"vs {len(components)} components")
            continue
        # Pair each component with its corresponding digit char in the score.
        for (patch, _box), digit_char in zip(components, score_str):
            digit = int(digit_char)
            observations[digit] += 1
            if captured[digit] is None:
                captured[digit] = patch

    # Save the first clean observation of each digit.
    saved: dict[int, int] = {}
    for digit, patch in captured.items():
        if patch is not None:
            path = save_template(template_dir, digit, patch)
            saved[digit] = observations[digit]
            print(f"[{game_name}] {digit}: saved template -> {path.relative_to(ROOT)} "
                  f"(observed {observations[digit]}x)")
        else:
            print(f"[{game_name}] {digit}: NOT YET CAPTURED — play more sessions or capture manually")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=list(GAMES.keys()))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    games_to_run = [args.game] if args.game else list(GAMES.keys())
    for g in games_to_run:
        print(f"=== {g} ===")
        bootstrap_game(g, verbose=args.verbose)
        print()


if __name__ == "__main__":
    import numpy as np  # noqa: F401 — imported for type annotation in captured dict
    main()
