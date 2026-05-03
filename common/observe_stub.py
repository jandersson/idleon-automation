"""Reusable 'watch me play' run() for minigames that don't yet have
real detectors / DB schemas.

Saves a frame snapshot per click into <minigame>/assets/observations/.
First-contact tool: play the game manually with this running, then
review the captured frames offline to figure out what's worth
detecting.

Each minigame's observe.py just re-exports run() with its own paths:

    from common.observe_stub import make_observe_run
    from minigames.foo.main import WINDOW_TITLE
    run = make_observe_run(__file__, WINDOW_TITLE)
"""
from pathlib import Path

from common.observer import ObserverConfig, run_observer, simple_frame_saver
from common.session_log import session_log


def make_observe_run(module_file: str, window_title: str, cooldown_seconds: float = 4.0):
    """Build a run() callable for a minigame's observe.py."""
    here = Path(module_file).parent
    out_dir = here / "assets" / "observations"
    logs_dir = here / "assets" / "logs"

    def run() -> None:
        with session_log(logs_dir) as log_path:
            print(f"Session log: {log_path}")
            print(
                f"OBSERVE — click on {window_title!r} to record snapshots.\n"
                f"  Frames saved to: {out_dir}\n"
                "  Overlay shows GO (green) / cooldown (red).\n"
                "  Move mouse to a screen corner to abort.\n"
            )
            n = run_observer(
                ObserverConfig(window_title=window_title,
                               cooldown_seconds=cooldown_seconds),
                simple_frame_saver(out_dir),
            )
            print(f"Captured {n} frames.")

    return run
