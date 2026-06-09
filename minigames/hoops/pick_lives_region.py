"""Click two corners around the Lives-Left digit; saves to regions.json as fractions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_region_picker
from minigames.hoops.main import WINDOW_TITLE

run = make_region_picker(Path(__file__).parent, WINDOW_TITLE, "lives", prompt="Pick the Lives-Left region (just the number digit). Capture starts in 3s.")

if __name__ == "__main__":
    run()
