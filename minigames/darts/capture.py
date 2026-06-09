"""Burst-capture frames of the darts minigame to assets/captures/.

Used for offline analysis of arm-swing dynamics: how long the release
pose template matches per swing, what the conf trajectory looks like
across consecutive frames, etc.

Usage:
  uv run darts-capture                          # default: 30 frames @ 100ms (3s)
  uv run darts-capture --interval 0.02          # bot-poll-rate sampling (50fps)
  uv run darts-capture --interval 0.02 --count 150   # 3s burst at bot poll rate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture_burst import capture_burst
from minigames.darts.main import WINDOW_TITLE


def run():
    capture_burst(
        WINDOW_TITLE,
        Path(__file__).parent / "assets" / "captures",
        label="darts minigame",
    )


if __name__ == "__main__":
    run()
