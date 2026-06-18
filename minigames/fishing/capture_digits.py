"""Dump isolated PTS digit glyphs from saved fishing score crops, to grow the
digit-template library the score reader matches against (#58 / #63 score-delta
catch detection).

The score reader (``score.py``) reads the "N PTS" counter by matching each
white-filled digit glyph against a per-digit template under
``assets/digit_templates/<d>.png``. The font is fixed, so each digit needs
capturing exactly once -- but the bootstrap frames only reached "0 PTS"
(0/2/6 were seeded from "0 PTS" / "26 BEST"), so 1/3/4/5/7/8/9 must come from a
live session that scores higher.

Workflow:
  1. Play a game with score-crop saving on:
       ``uv run fishing --save-frames``
     The bot saves each cast's PTS crop (before + after the cast) to
     ``assets/captures/botrun_<stamp>/score_*.png`` -- as the score climbs these
     cover digits 1-9.
  2. Run ``fishing-capture-digits`` (optionally pass a frames directory; default
     is the most recent ``botrun_*``). It binarizes each crop with the SAME
     white-fill binarizer the live reader uses, isolates the digit glyphs, writes
     each as a native binary patch to ``assets/digit_capture/``, and builds an
     annotated contact sheet ``assets/digit_capture/_contact_sheet.png``.
  3. Open the contact sheet, identify each glyph, and copy the matching patch
     file to ``assets/digit_templates/<digit>.png`` (the patch that is a clean
     "5" -> ``digit_templates/5.png``). Commit the new templates -- the reader
     picks them up on next run.

This stays a manual label step on purpose: auto-labeling from the read score
would silently poison the committed library whenever a read is wrong. Full-window
frames are also accepted (the cast bar is detected and the score region cropped),
so a ``botrun`` of raw frames works too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

from common.score_digits import binarize_white_fill, digit_components
from minigames.fishing.detector import find_cast_bar
from minigames.fishing.score import score_crop

_HERE = Path(__file__).parent
CAPTURES_DIR = _HERE / "assets" / "captures"
CAPTURE_DIR = _HERE / "assets" / "digit_capture"
PREVIEW_SCALE = 12
# A saved score crop is small (~19px tall); anything taller is a full-window
# frame we crop the score region out of (detect the cast bar first).
CROP_MAX_HEIGHT = 60


def _latest_botrun() -> Path | None:
    """The most recent ``captures/botrun_*`` dir (timestamped names sort)."""
    runs = sorted(p for p in CAPTURES_DIR.glob("botrun_*") if p.is_dir())
    return runs[-1] if runs else None


def _score_crop_of(img: np.ndarray) -> np.ndarray | None:
    """The score crop from a saved image: used directly if it's already a small
    score crop, else cropped from a full-window frame via the detected cast bar."""
    if img.shape[0] <= CROP_MAX_HEIGHT:
        return img
    return score_crop(img, find_cast_bar(img))


def run(frames_dir: Path | None = None) -> None:
    if len(sys.argv) > 1:
        frames_dir = Path(sys.argv[1])
    frames_dir = frames_dir or _latest_botrun()
    if frames_dir is None or not frames_dir.is_dir():
        print(f"No frames directory ({frames_dir}). Run `uv run fishing --save-frames` "
              f"first (it saves score_*.png crops under assets/captures/botrun_*).")
        return
    # Prefer the bot's saved score crops; fall back to every PNG in the dir.
    frames = sorted(frames_dir.glob("score_*.png")) or sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"No .png frames in {frames_dir}.")
        return
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    tiles: list[np.ndarray] = []
    n_glyphs = 0
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        crop = _score_crop_of(img)
        if crop is None or crop.size == 0:
            continue
        for patch, (x, _y, w, _h) in digit_components(crop, binarize_white_fill):
            stem = f"{frame_path.stem}_x{x:03d}_w{w}"
            cv2.imwrite(str(CAPTURE_DIR / f"{stem}.png"), patch)
            preview = cv2.resize(patch, (patch.shape[1] * PREVIEW_SCALE, patch.shape[0] * PREVIEW_SCALE),
                                 interpolation=cv2.INTER_NEAREST)
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            label = np.full((16, preview.shape[1], 3), 30, np.uint8)
            cv2.putText(label, stem[:26], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            tiles.append(np.vstack([label, preview]))
            n_glyphs += 1

    if not tiles:
        print("No digit glyphs found -- are the frames PTS score crops (or "
              "full-window frames with the cast bar visible)?")
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
