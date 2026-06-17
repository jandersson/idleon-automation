"""Click two corners around the "N PTS" number; saves to regions.json as
fractions of the current window size so it survives resizing.

The catching HUD is anchored to the player's minigame position, so the
default score region in regions.json is only a starting point -- re-pick it
for your setup. Bracket the digits (and a little margin for the score
growing to two digits); the trailing "PTS" label is tolerated, the reader
drops it. Stay left of "BEST".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_region_picker
from minigames.catching.main import WINDOW_TITLE

run = make_region_picker(Path(__file__).parent, WINDOW_TITLE, "score")

if __name__ == "__main__":
    run()
