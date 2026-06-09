"""Factories for the two standard one-off picker scripts every minigame
carries (#36).

Before this module each minigame copy-pasted ~30-45 lines per picker with
only the region name, prompt text, and WINDOW_TITLE differing. A picker
script now declares those three things and delegates here. Pickers with
genuinely different flows (mining's offline-first play picker, darts'
release-pose cropper) stay hand-written in their minigame folder.
"""
import time
from pathlib import Path

import cv2

from common.capture import grab_region
from common.region_picker import pick_region
from common.regions import save_region
from common.window import get_bounds


def make_region_picker(
    minigame_dir: Path,
    window_title: str,
    region_name: str,
    prompt: str | None = None,
):
    """Build a run() that picks a rectangle on the live window and saves
    it to the minigame's regions.json as fractions of the window size
    (so it survives resizing)."""

    def run():
        print(prompt or f"Pick the {region_name} region. Capture starts in 3s.")
        time.sleep(3)
        region = pick_region(window_title=window_title, region_name=region_name)
        if region is None:
            return
        _, _, win_w, win_h = get_bounds(window_title)
        path = save_region(minigame_dir, region_name, region, win_w, win_h)
        print(f"Saved {region_name} region (as fractions of {win_w}x{win_h}) to {path}: {region}")

    return run


def make_template_picker(
    window_title: str,
    out_path: Path,
    element_name: str = "game over",
):
    """Build a run() that crops a template image from the live window.

    Capturing through the live pipeline (grab_region + BGRA->BGR) matters:
    cropping from a manual screenshot mismatches at match time because of
    color-space and scaling differences.
    """

    def run():
        print(f"Bring the {element_name} element into view. Capture starts in 3s.")
        time.sleep(3)
        left, top, width, height = get_bounds(window_title)
        print(f"Window at ({left}, {top}) {width}x{height}")
        region = pick_region(window_title=window_title, region_name=element_name)
        if region is None:
            return
        # Re-grab now so the crop matches what the user just clicked on.
        frame = grab_region(left, top, width, height)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        crop = bgr[region["top"]:region["top"] + region["height"],
                   region["left"]:region["left"] + region["width"]]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop)
        print(f"Wrote {element_name} template ({crop.shape[1]}x{crop.shape[0]}) to {out_path}")

    return run
