"""Tkinter launcher for the Idleon bots (per-tab package, issue #24).

Public entry point: `run()` (wired as the `idleon` console script). The
Launcher shell lives in app.py; each tab is its own module
(bots_tab/setup_tab/frames_tab/sql_tab) with a `build(parent, app)`
function, sharing state via the `app` reference. theme.py holds the dark
palette + ttk styling; config.py holds the minigame registry.
"""
from ui.launcher.app import Launcher, run

__all__ = ["Launcher", "run"]


if __name__ == "__main__":
    run()
