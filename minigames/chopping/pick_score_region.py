"""Click two corners around the live "N PTS" counter (left of the chop
bar overlay). Saves to regions.json as fractions. The bot OCRs it for
per-chop score ground truth — the increments-based estimate can't catch
e.g. a wrong gold-pays-double assumption, the counter can.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_region_picker
from minigames.chopping.main import WINDOW_TITLE

run = make_region_picker(
    Path(__file__).parent, WINDOW_TITLE, "score",
    prompt="Pick the 'N PTS' counter region (digits only if possible). Capture starts in 3s.",
)

if __name__ == "__main__":
    run()
