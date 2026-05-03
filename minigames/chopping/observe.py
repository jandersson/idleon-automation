"""Watch me play chopping — saves one frame per click for offline review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.observe_stub import make_observe_run
from minigames.chopping.main import WINDOW_TITLE

run = make_observe_run(__file__, WINDOW_TITLE)


if __name__ == "__main__":
    run()
