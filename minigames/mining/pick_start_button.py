"""Pick a bounding box around the in-game 'Start Game' button.

The trace script clicks the center of this region to start an attempt
deterministically. Always grabs a fresh live screenshot — the Start
prompt is only visible in a specific game state, so saved captures
typically don't show it.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.region_picker import pick_region
from common.regions import save_region
from common.window import get_bounds
from minigames.mining.main import WINDOW_TITLE

MINIGAME_DIR = Path(__file__).parent
CAPTURES_DIR = MINIGAME_DIR / "assets" / "captures"


def run():
    parser = argparse.ArgumentParser(description="Pick start_button region from live grab")
    parser.add_argument("--startup", type=float, default=5.0,
                        help="seconds before screenshot is taken (default 5)")
    args = parser.parse_args()

    print(f"Bring the mining Start prompt on-screen. Live grab in {args.startup:.0f}s.")
    time.sleep(args.startup)

    # Grab once and use those exact dimensions both for the picker and for
    # the saved fraction reference. Calling get_bounds twice (once inside
    # pick_region, once for save_region) can disagree if focus changes
    # between the calls and matches a differently-sized window.
    left, top, win_w, win_h = get_bounds(WINDOW_TITLE)
    print(f"Captured {WINDOW_TITLE!r}: ({left},{top}) {win_w}x{win_h}")
    frame = grab_region(left, top, win_w, win_h)
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_path = CAPTURES_DIR / f"start_prompt_{stamp}.png"
    cv2.imwrite(str(snap_path), bgr)
    print(f"Saved snapshot for picking: {snap_path.name}")

    region = pick_region(region_name="start_button", image_path=snap_path)
    if region is None:
        return
    path = save_region(MINIGAME_DIR, "start_button", region, win_w, win_h)
    print(f"Saved start_button region (as fractions of {win_w}x{win_h}) to {path}: {region}")


if __name__ == "__main__":
    run()
