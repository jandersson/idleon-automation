"""Dump isolated PTS digit glyphs from saved frames, to grow the template
library (issue #47 / #52 score logging).

The score reader (``score.py``) matches each digit against a per-digit
template under ``assets/digit_templates/<d>.png``. The font is fixed, so
each digit needs capturing exactly once -- but the observe frames only
reached "1 PTS", so 2-9 must be captured from a live session that scores
higher.

Workflow:
  1. Play/observe a game that reaches a higher score (``catching-observe``
     saves one full-window frame per click to ``assets/observations/``).
  2. Run ``catching-capture-digits`` (optionally pass a frames directory).
     It crops the ``score`` region from each frame, isolates the digit
     glyphs the live reader would see, writes each as a native binary patch
     to ``assets/digit_capture/``, and builds an annotated contact sheet
     ``assets/digit_capture/_contact_sheet.png``.
  3. Open the contact sheet, identify each glyph, and copy the matching
     patch file to ``assets/digit_templates/<digit>.png`` (e.g. the patch
     that is a clean "5" -> ``digit_templates/5.png``). Commit the new
     templates -- the reader picks them up on next run.

This stays a manual label step on purpose: auto-labeling from score
increments would silently poison the committed library if an increment were
miscounted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

from common.regions import get_region
from minigames.catching.score import digit_components

_HERE = Path(__file__).parent
OBSERVATIONS_DIR = _HERE / "assets" / "observations"
CAPTURE_DIR = _HERE / "assets" / "digit_capture"
PREVIEW_SCALE = 10


def _score_crop(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Crop the score region from a full-window BGR frame using its size."""
    h, w = frame_bgr.shape[:2]
    region = get_region(_HERE, "score", w, h)
    if region is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x0, y0 = max(0, region["left"]), max(0, region["top"])
    x1 = min(w, region["left"] + region["width"])
    y1 = min(h, region["top"] + region["height"])
    return gray[y0:y1, x0:x1]


def run(frames_dir: Path | None = None) -> None:
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (frames_dir or OBSERVATIONS_DIR)
    if not frames_dir.is_dir():
        print(f"No frames directory at {frames_dir}. Run catching-observe first.")
        return
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"No .png frames in {frames_dir}.")
        return
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    tiles: list[np.ndarray] = []
    n_glyphs = 0
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        crop = _score_crop(frame)
        if crop is None:
            print("No 'score' region in regions.json. Pick it with "
                  "catching-pick-score-region first.")
            return
        for patch, (x, _y, w, _h) in digit_components(crop):
            stem = f"{frame_path.stem}_x{x:03d}_w{w}"
            cv2.imwrite(str(CAPTURE_DIR / f"{stem}.png"), patch)
            preview = cv2.resize(patch, (patch.shape[1] * PREVIEW_SCALE, patch.shape[0] * PREVIEW_SCALE),
                                 interpolation=cv2.INTER_NEAREST)
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            label = np.full((16, preview.shape[1], 3), 30, np.uint8)
            cv2.putText(label, stem[:24], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            tiles.append(np.vstack([label, preview]))
            n_glyphs += 1

    if not tiles:
        print("No digit glyphs found in any frame -- is the score region "
              "pointed at the 'N PTS' number? Re-pick with catching-pick-score-region.")
        return
    width = max(t.shape[1] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 4, 4, 4, width - t.shape[1] + 4, cv2.BORDER_CONSTANT, value=(60, 60, 60))
              for t in tiles]
    sheet = np.vstack(padded)
    sheet_path = CAPTURE_DIR / "_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"Wrote {n_glyphs} glyph patches + {sheet_path}")
    print(f"Inspect {sheet_path.name}, then copy each clean glyph's patch to "
          f"assets/digit_templates/<digit>.png and commit.")


if __name__ == "__main__":
    run()
