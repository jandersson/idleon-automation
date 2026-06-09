"""Burst-capture frames of the hoops minigame for template extraction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture_burst import capture_burst
from minigames.hoops.main import WINDOW_TITLE


def run():
    capture_burst(
        WINDOW_TITLE,
        Path(__file__).parent / "assets" / "captures",
        post_hint="Scrub through them, pick the clearest cue, crop it out, save as assets/indicator.png",
        label="hoops minigame",
    )


if __name__ == "__main__":
    run()
