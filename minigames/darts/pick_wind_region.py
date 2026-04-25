"""Click two corners of the wind indicator in the dart UI.

Prints a WIND_REGION_REL dict to paste into main.py. Tight crop around just
the wind text or arrow — exclude the surrounding "Wind:" label if possible
so template matching is sensitive to the actual changing values.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from minigames.darts.main import WINDOW_TITLE


def run():
    print("Pick the wind-indicator region. Bring the darts minigame into view; capture starts in 3s.")
    time.sleep(3)
    region = pick_region(window_title=WINDOW_TITLE, region_name="wind")
    if region is None:
        return
    print()
    print("Paste into minigames/darts/main.py:")
    print(f"WIND_REGION_REL: dict | None = {region}")


if __name__ == "__main__":
    run()
