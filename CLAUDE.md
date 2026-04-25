# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Screen-reading bots for the game **Idleon**. Each bot grabs a region of the screen, runs OpenCV against it, and fires synthetic clicks via `pyautogui`. Windows-only in practice (depends on `pygetwindow`).

The README covers per-minigame run commands and tuning knobs — don't duplicate that here. This file documents the cross-cutting patterns.

## Setup

`pip install -e .` or `uv sync` — both work, `uv.lock` is checked in. Python 3.11+.

The project has no tests, linter config, or formatter. Don't add them unless asked.

## Architecture

### Layering

```
common/             IO layer — screen capture, clicks, window lookup
minigames/<name>/   per-minigame bot, one folder each
```

Every minigame folder follows the same quartet:

- `main.py` — main loop with a `run()` entry point. **Also the config file** for that minigame: holds `WINDOW_TITLE`, region constants, tuning knobs. Other scripts in the folder import these from `main`.
- `detector.py` — pure CV functions, no IO. Takes a frame, returns coordinates / classifications.
- `capture.py` or `calibrate.py` — optional one-off tooling that writes debug images to disk. Used to gather templates (hoops) or visualize HSV masks (chopping).
- `assets/` — templates, captures, or calibration output.

When adding a new minigame, mirror this structure and register entry points in `pyproject.toml`'s `[project.scripts]`.

### Coordinate convention

All region constants in source are **window-relative**, not screen-absolute. The window's screen position is resolved at runtime via `common.window.get_bounds(WINDOW_TITLE)`, which matches by case-insensitive title substring. Capture and click coordinates are computed by adding the window's `(left, top)` each tick — so the bot survives the user moving the game window mid-run.

This also means: never hardcode screen coordinates. Always express positions relative to the window.

### Frame format gotcha

`mss.grab()` returns BGRA. OpenCV operations expect BGR. Detectors do `cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)` as their first step — preserve this when adding new detectors.

### Detection styles

Two approaches are in use; pick whichever fits the visual cue:

- **Template matching** (`hoops`): `cv2.matchTemplate` with `TM_CCOEFF_NORMED`, region-restricted (right half for hoop, left half for platform) to cut false positives. Templates must be captured *through the same pipeline as the live bot* — that's what `hoops-capture` is for. Cropping a template from a manual screenshot will mismatch because of color-space and scaling differences.
- **HSV color masking** (`chopping`): `cv2.inRange` on HSV channels, with priority resolution (gold > green > red). Always tune via the `<minigame>-calibrate` script, which dumps per-range mask overlays to `calibration/`.

### Entry points and the sys.path dance

Each `main.py` and tooling script starts with:

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

This makes `from common...` and `from minigames...` work whether launched as a module, as a script, or via the `[project.scripts]` console_scripts entry. Keep it when adding new entry-point scripts.

### Safety

`pyautogui.FAILSAFE = True` is set globally in `common/input.py`. Slamming the mouse into any screen corner aborts. Every `main.run()` opens with a 2-second sleep so the user can switch to the game window before clicks start. Preserve both conventions in new bots.

`common.input.click` adds ±3px positional jitter and `random_delay` adds 80–200ms by default — don't remove the randomization, it's part of looking non-bot-like to the game.
