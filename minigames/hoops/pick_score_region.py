"""Click two corners around the score number; saves to regions.json.

main.py picks it up automatically on next run — no copy-paste needed.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from common.regions import save_region
from minigames.hoops.main import WINDOW_TITLE

MINIGAME_DIR = Path(__file__).parent


def run():
    print("Pick the score region. Capture starts in 3s.")
    time.sleep(3)
    region = pick_region(window_title=WINDOW_TITLE, region_name="score")
    if region is None:
        return
    path = save_region(MINIGAME_DIR, "score", region)
    print(f"Saved score region to {path}: {region}")


if __name__ == "__main__":
    run()
