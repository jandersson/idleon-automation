"""Click two corners around the wind indicator; saves as fractions to regions.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_region_picker
from minigames.darts.main import WINDOW_TITLE

run = make_region_picker(Path(__file__).parent, WINDOW_TITLE, "wind")

if __name__ == "__main__":
    run()
