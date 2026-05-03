"""Watch me play catching — saves one frame per click for offline review.

Catching is the least-developed minigame; this is the right starting
point for figuring out what to detect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.observe_stub import make_observe_run
from minigames.catching.main import WINDOW_TITLE

run = make_observe_run(__file__, WINDOW_TITLE)


if __name__ == "__main__":
    run()
