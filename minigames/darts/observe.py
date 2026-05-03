"""Watch me play darts — saves one frame per click for offline review.

Use this when first approaching the minigame: play through a full
trial manually, then look at assets/observations/ to figure out what
visual features to detect (release pose, score region, wind indicator,
etc.).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.observe_stub import make_observe_run
from minigames.darts.main import WINDOW_TITLE

run = make_observe_run(__file__, WINDOW_TITLE)


if __name__ == "__main__":
    run()
