"""Crop the game-over element from the live game window and save as template.

Run this when the game-over screen is currently visible. Click two corners
around the stable on-screen text/element that identifies the end-of-game
state. Saves to assets/game_over.png so find_game_over can use it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.pick_tools import make_template_picker
from minigames.chopping.main import WINDOW_TITLE

run = make_template_picker(WINDOW_TITLE, Path(__file__).parent / "assets" / "game_over.png")

if __name__ == "__main__":
    run()
