"""Every minigame's main.py must call common.input.check_failsafe in its main
loop. The fail-safe is the only way the user can abort a runaway bot — a
minigame that forgets to wire it in is a foot-gun. Catching this at test time
is much cheaper than discovering it while a bot is spamming clicks.

We don't try to verify the call is *inside* the loop (AST-walking the loop
body is brittle). A top-level `check_failsafe()` call would still pass —
acceptable, since the realistic regression we're guarding against is
"forgot to import/call it at all."
"""
import ast
from pathlib import Path

import pytest


MINIGAMES_DIR = Path(__file__).parent.parent / "minigames"


def _minigame_mains() -> list[Path]:
    return sorted(p for p in MINIGAMES_DIR.glob("*/main.py") if p.is_file())


@pytest.mark.parametrize("main_path", _minigame_mains(), ids=lambda p: p.parent.name)
def test_main_calls_check_failsafe(main_path: Path):
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    found = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check_failsafe"
        for node in ast.walk(tree)
    )
    assert found, (
        f"{main_path.relative_to(MINIGAMES_DIR.parent)} does not call "
        f"check_failsafe(). Import it from common.input and call it from "
        f"each iteration of the main loop so the corner-snap abort works."
    )
