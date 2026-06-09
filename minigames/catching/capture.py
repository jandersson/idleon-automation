"""Burst-capture frames of the catching minigame for template extraction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture_burst import capture_burst
from minigames.catching.main import WINDOW_TITLE


def run():
    capture_burst(
        WINDOW_TITLE,
        Path(__file__).parent / "assets" / "captures",
        label="catching minigame",
    )


if __name__ == "__main__":
    run()
