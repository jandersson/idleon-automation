"""Click two corners around the score number; saves to regions.json as
fractions of the current window size so it survives resizing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_region_picker
from minigames.hoops.main import WINDOW_TITLE

run = make_region_picker(Path(__file__).parent, WINDOW_TITLE, "score")

if __name__ == "__main__":
    run()
