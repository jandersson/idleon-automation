"""Click two corners around the fishing minigame play area (the water where
fish/mines appear and the lure casts). Saves to regions.json as fractions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_region_picker
from minigames.fishing.main import WINDOW_TITLE

run = make_region_picker(
    Path(__file__).parent, WINDOW_TITLE, "play",
    prompt="Pick the fishing play region (the water). Capture starts in 3s.",
)

if __name__ == "__main__":
    run()
