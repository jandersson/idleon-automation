"""Click two corners around the LEAF'S TRACK (the strip above the bar where
the leaf scrolls back and forth). Saves to regions.json as fractions.

The leaf and bar are different regions: leaf for X-position detection,
bar for zone-color lookup at that X.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_region_picker
from minigames.chopping.main import WINDOW_TITLE

run = make_region_picker(
    Path(__file__).parent, WINDOW_TITLE, "leaf",
    prompt="Pick the LEAF region — the strip ABOVE the bar where the leaf scrolls. Capture starts in 3s.",
)

if __name__ == "__main__":
    run()
