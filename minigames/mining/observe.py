"""Watch me play mining — saves one frame per click for offline review.

Each click (jump or slam) writes a snapshot to assets/observations/. Use
this to collect ground-truth frames showing the cart at the instant of
human decisions: airborne start, mid-jump, slam-down, hit-ore, miss-into-pit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.observe_stub import make_observe_run
from minigames.mining.main import WINDOW_TITLE

run = make_observe_run(__file__, WINDOW_TITLE)


if __name__ == "__main__":
    run()
