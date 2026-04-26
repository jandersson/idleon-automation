"""Click two corners around just the score number for darts.

Tight crop on JUST the digits (no 'Score:' label, no surrounding decorations)
so digit-segmentation/matching has a clean strip to work with.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from minigames.darts.main import WINDOW_TITLE


def run():
    print("Pick the score region (just the number digits). Capture starts in 3s.")
    time.sleep(3)
    region = pick_region(window_title=WINDOW_TITLE, region_name="score")
    if region is None:
        return
    print()
    print("Paste into minigames/darts/main.py, replacing SCORE_REGION_REL:")
    print(f"SCORE_REGION_REL: dict | None = {region}")


if __name__ == "__main__":
    run()
