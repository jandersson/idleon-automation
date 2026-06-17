"""Watch me play fishing — saves one frame per click for offline review.

Fishing is freshly scaffolded; this is the starting point for verifying
what the detector sees (fish colours, mines, the lure) on real frames
before trusting the bot to cast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.observe_stub import make_observe_run
from minigames.fishing.main import WINDOW_TITLE

run = make_observe_run(__file__, WINDOW_TITLE)


if __name__ == "__main__":
    run()
