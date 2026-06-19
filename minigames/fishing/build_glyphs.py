"""Build the masked-ZNCC glyph templates the fishing score reader matches against
(#82 -- background-invariant PTS reading).

The binarize-then-match reader (``digit_templates/`` + ``binarize_white_fill``)
floods on pale-desaturated backgrounds (the new water area), so the reader now
matches each digit by its GLYPH PATTERN via masked ZNCC -- the same move that
made fish detection biome-invariant (#75). This tool builds, per digit 0-9:

  * ``digit_glyphs/<d>.png`` -- a MEDIAN grayscale glyph (bright fill + dark
    outline/interior), stacked over many corpus instances to suppress noise;
  * ``digit_glyphs/<d>_mask.png`` -- the ink mask (the bright fill dilated 1px to
    grab the dark outline), so every masked pixel is intrinsic to the rendered
    glyph and the background behind the overlay is ignored.

It bootstraps the labels from the EXISTING white-fill reader on the dock corpus
(where that reader works): each ``score_c*.png`` crop is binarized, its digit
components walked left-to-right and matched against ``digit_templates/`` to label
them, and the grayscale glyph at each labelled bbox (pad 1) collected. The
masked-ZNCC reader these feed is then validated to be background-invariant
(reads the pale-water beach frame) and zero-regression on the dock corpus by
``scripts/validate_fishing_glyphs2.py`` -- run that after rebuilding.

Usage (writes/overwrites ``assets/digit_glyphs/``):

    uv run fishing-build-glyphs            # default: all assets/captures/botrun_*
    uv run fishing-build-glyphs <dir>      # a specific frames dir

This is a build step, not a live path; it reads only saved crops. The glyph
templates are tiny and committed to git (the reader loads them at startup).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

from common.score_digits import (
    binarize_white_fill, digit_components, load_digit_templates, _canon, match_digit,
)

_HERE = Path(__file__).parent
CAPTURES_DIR = _HERE / "assets" / "captures"
DIGIT_TEMPLATES_DIR = _HERE / "assets" / "digit_templates"   # old binary, for labelling
GLYPH_DIR = _HERE / "assets" / "digit_glyphs"                # new grayscale + mask
CROP_MAX_HEIGHT = 60   # taller inputs are full frames; saved score crops are ~12-19px
GLYPH_FILL_V_MIN = 190  # the neutral-white digit core, reliable in all backgrounds


def _collect_instances(frames: list[Path]) -> dict[int, list[np.ndarray]]:
    """Label leading digits with the white-fill reader; return digit -> list of
    grayscale glyph bboxes (pad 1)."""
    labels = {d: [_canon(t) for t in v]
              for d, v in load_digit_templates(DIGIT_TEMPLATES_DIR).items()}
    inst: dict[int, list[np.ndarray]] = defaultdict(list)
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[0] > CROP_MAX_HEIGHT:   # a full frame slipped in: skip (need the bar)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for patch, (x, y, w, h) in digit_components(img, binarize_white_fill):
            digit, _ = match_digit(patch, labels)
            if digit is None:
                break                        # stop at the first non-digit (the label)
            y0, y1 = max(0, y - 1), min(gray.shape[0], y + h + 1)
            x0, x1 = max(0, x - 1), min(gray.shape[1], x + w + 1)
            inst[digit].append(gray[y0:y1, x0:x1])
    return inst


def _build(instances: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Median grayscale glyph + ink mask at the modal native (padded) size."""
    size = Counter(g.shape for g in instances).most_common(1)[0][0]
    stack = np.array([g for g in instances if g.shape == size], dtype=np.float32)
    glyph = np.median(stack, axis=0).astype(np.uint8)
    fill = (glyph >= GLYPH_FILL_V_MIN).astype(np.uint8)
    mask = cv2.dilate(fill, np.ones((3, 3), np.uint8), iterations=1) * 255
    return glyph, mask


def run(frames_dir: Path | None = None) -> None:
    if len(sys.argv) > 1:
        frames_dir = Path(sys.argv[1])
    if frames_dir is not None:
        frames = sorted(frames_dir.glob("score_*.png")) or sorted(frames_dir.glob("*.png"))
    else:
        frames = sorted(CAPTURES_DIR.glob("botrun_*/score_*.png"))
    if not frames:
        print(f"No score crops found (looked under {frames_dir or CAPTURES_DIR}). "
              f"Run `uv run fishing --save-frames` first.")
        return

    inst = _collect_instances(frames)
    missing = [d for d in range(10) if not inst.get(d)]
    if missing:
        print(f"WARNING: no instances for digit(s) {missing} -- the reader will "
              f"return None on numbers needing them. Capture more frames.")

    GLYPH_DIR.mkdir(parents=True, exist_ok=True)
    for digit in range(10):
        if not inst.get(digit):
            continue
        glyph, mask = _build(inst[digit])
        cv2.imwrite(str(GLYPH_DIR / f"{digit}.png"), glyph)
        cv2.imwrite(str(GLYPH_DIR / f"{digit}_mask.png"), mask)
        print(f"  digit {digit}: {glyph.shape} from {len(inst[digit])} instances, "
              f"mask px {int((mask > 0).sum())}")
    print(f"Wrote glyph templates to {GLYPH_DIR}. Validate with "
          f"`uv run --no-sync python scripts/validate_fishing_glyphs2.py`.")


if __name__ == "__main__":
    run()
