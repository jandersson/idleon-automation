"""Save play-region frames to assets/captures/ for template extraction.

The fishing detector needs a few sprite templates that can't be guessed
from colour: the lure (assets/lure.png — for landing detection), the
'play minigame' prompt (assets/play_button.png), and the game-over screen
(assets/game_over.png). Run this while playing, then crop those sprites
out of the saved frames.

Run:  fishing-capture     (Ctrl-C to stop)
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2

from common.capture import grab_region
from common.input import check_failsafe
from common.regions import get_region
from common.window import get_bounds, WindowNotFoundError
from minigames.fishing.main import WINDOW_TITLE

_HERE = Path(__file__).parent
OUT_DIR = _HERE / "assets" / "captures"
INTERVAL_S = 1.0


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fishing capture — saving a play-region frame every {INTERVAL_S:.0f}s "
          f"to {OUT_DIR}. Ctrl-C to stop.")
    time.sleep(2)
    n = 0
    while True:
        check_failsafe()
        try:
            left, top, w, h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue
        play = get_region(_HERE, "play", w, h)
        if play is None:
            print("No 'play' region yet — run fishing-pick-play-region first.")
            time.sleep(2)
            continue
        frame = grab_region(left + play["left"], top + play["top"], play["width"], play["height"])
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        path = OUT_DIR / f"frame_{datetime.now():%H%M%S}_{n:03d}.png"
        cv2.imwrite(str(path), bgr)
        n += 1
        if n % 10 == 0:
            print(f"  saved {n} frames")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    run()
