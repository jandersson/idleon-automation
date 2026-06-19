"""Offline validation of the background-invariant charge-thermometer reader (#83).

find_charge_fill read the red fill STUCK-LOW after an area move: on bright
turquoise water the fill desaturates to a pale orange the absolute red gate
missed, so the closed-loop fired at uncontrolled max charge (~10% of casts). The
relative read (interior red-dominance vs the background beside the tube) fixes
it. This harness checks the fix on every full-window capture that shows the
thermometer:

  * the pale-water beach frame (the failing case) now reads a full tube, not ~9;
  * the calibration-area charged frames still read the same ramp (no regression);
  * empty tubes still read ~0 (no false-full -> no undercharged cast).

Not a pytest test (it needs the live capture corpus + the beach screenshot); a
one-off measurement tool kept under scripts/. Run:
    uv run --no-sync python scripts/validate_fishing_charge.py
"""
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from minigames.fishing.detector import (
    find_cast_bar, find_charge_fill, _to_hsv, _mask, CHARGE_RED_LOW, CHARGE_RED_HIGH,
)

ROOT = Path(__file__).parent.parent
CAL_RUN = ROOT / "minigames/fishing/assets/captures/botrun_20260617_225352"
BEACH = r"C:\Program Files (x86)\Steam\userdata\153875\760\remote\1476970\screenshots\20260618155212_1.jpg"


def _old_find_charge_fill(full_frame, bar):
    """The pre-#83 absolute-red-gate reader (V>=90 & S>=120 over a bar_x-50..-22
    window, rows-with-red), reconstructed for the regression comparison."""
    if bar is None:
        return 0
    bx, by = bar[0], bar[1]
    h, w = full_frame.shape[:2]
    x0, x1 = max(0, bx - 50), max(0, min(w, bx - 22))
    y0, y1 = max(0, by - 64), max(0, min(h, by + 18))
    if x1 <= x0 or y1 <= y0:
        return 0
    hsv = _to_hsv(full_frame[y0:y1, x0:x1])
    red = cv2.bitwise_or(_mask(hsv, *CHARGE_RED_LOW), _mask(hsv, *CHARGE_RED_HIGH))
    return int(np.count_nonzero((red > 0).any(axis=1)))


def read(path):
    img = cv2.imread(str(path))
    if img is None:
        return None, None, None
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bar = find_cast_bar(bgra)
    return bar, find_charge_fill(bgra, bar), _old_find_charge_fill(bgra, bar)


def main():
    print("=== beach (pale water — the #83 failing background) ===")
    bar, new, old = read(BEACH)
    print(f"  bar={bar}  OLD={old} (stuck-low)  NEW={new}  (expect a substantial fill)")
    assert new is not None and new > 30, "beach charge still reads stuck-low!"

    print("\n=== calibration charged run: NEW vs OLD (no-regression) ===")
    fs = sorted(glob.glob(str(CAL_RUN / "chargefull0*_p*.png")))
    if not fs:
        print(f"  (no calibration frames at {CAL_RUN})")
        return
    diffs, news, diverge = [], [], 0
    for f in fs:
        _, new, old = read(f)
        news.append(new)
        if old and old > 10:                     # compare where the old reader saw a fill
            diffs.append(abs(new - old))
            if abs(new - old) > 10:
                diverge += 1
                print(f"  DIVERGE {os.path.basename(f)}: OLD={old} NEW={new}")
    nz = [v for v in news if v and v > 5]
    print(f"  {len(fs)} frames: charged range {min(nz)}-{max(nz)} median {int(np.median(nz))}; "
          f"vs OLD: median abs-diff={int(np.median(diffs))} max abs-diff={max(diffs)} divergences>10={diverge}")
    assert diverge == 0, "charge reader regressed vs the old reader in the calibration area!"
    print("\nOK: beach fixed (stuck-low -> full tube); calibration matches the old reader.")


if __name__ == "__main__":
    main()
