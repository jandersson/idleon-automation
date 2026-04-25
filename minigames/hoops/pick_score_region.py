import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from minigames.hoops.main import WINDOW_TITLE


def run():
    print("Pick the score region. Bring the hoops minigame into view; capture starts in 3s.")
    time.sleep(3)
    region = pick_region(WINDOW_TITLE, region_name="score")
    if region is None:
        return
    print()
    print("Paste this into minigames/hoops/main.py replacing SCORE_REGION_REL:")
    print(f"SCORE_REGION_REL: dict | None = {region}")


if __name__ == "__main__":
    run()
