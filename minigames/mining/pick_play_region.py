"""Pick the play region from the most recent captured frame.

Run `mining-capture` first during a live attempt, then run this offline —
no second attempt needed. Falls back to live capture if no frames exist.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from common.regions import save_region
from common.window import get_bounds
from minigames.mining.main import WINDOW_TITLE

MINIGAME_DIR = Path(__file__).parent
CAPTURES_DIR = MINIGAME_DIR / "assets" / "captures"


def _latest_capture() -> Path | None:
    if not CAPTURES_DIR.exists():
        return None
    pngs = sorted(CAPTURES_DIR.glob("capture_*.png"))
    return pngs[-1] if pngs else None


def run():
    latest = _latest_capture()
    if latest is not None:
        print(f"Picking from latest capture: {latest.name}")
        region = pick_region(region_name="play", image_path=latest)
        if region is None:
            return
        # The capture was the full game window, so its dimensions are the
        # window dims — read them off the image directly.
        import cv2
        img = cv2.imread(str(latest))
        win_h, win_w = img.shape[:2]
    else:
        print("No captures yet — falling back to live window grab. Capture starts in 3s.")
        time.sleep(3)
        region = pick_region(window_title=WINDOW_TITLE, region_name="play")
        if region is None:
            return
        _, _, win_w, win_h = get_bounds(WINDOW_TITLE)

    path = save_region(MINIGAME_DIR, "play", region, win_w, win_h)
    print(f"Saved play region (as fractions of {win_w}x{win_h}) to {path}: {region}")


if __name__ == "__main__":
    run()
