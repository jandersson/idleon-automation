# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Screen-reading bots for the game **Idleon**. Each bot grabs a region of the screen, runs OpenCV against it, and fires synthetic clicks via `pyautogui`. Windows-only in practice (depends on `pygetwindow`).

The README covers per-minigame run commands and tuning knobs — don't duplicate that here. This file documents the cross-cutting patterns.

## Workflow

**Always commit changes.** After completing a logical unit of work, commit it without waiting to be asked. Split unrelated changes into separate commits. Pushing follows the daytime-hours rule from user memory; committing has no time gate.

## Setup

`pip install -e .` or `uv sync` — both work, `uv.lock` is checked in. Python 3.11+.

There's a small pytest suite under `tests/`. `uv run pytest` runs it. Aimed at the pure-logic helpers (regions.json round-trips, multi-scale template matching against synthetic images, chopping zone lookup, score-diff binarization, hoops offset interpolation). No CV-against-real-game-frames tests — those are inherently visual, calibrated by the user, and don't generalize. Keep tests fast and self-contained; don't pull live screen captures.

No linter or formatter config. Don't add them unless asked.

## Architecture

### Layering

```
common/             IO layer — screen capture, clicks, window lookup
minigames/<name>/   per-minigame bot, one folder each
ui/                 user-facing UI (launcher, future dashboards)
```

Every minigame folder follows the same quartet:

- `main.py` — main loop with a `run()` entry point. **Also the config file** for that minigame: holds `WINDOW_TITLE`, region constants, tuning knobs. Other scripts in the folder import these from `main`.
- `detector.py` — pure CV functions, no IO. Takes a frame, returns coordinates / classifications.
- `capture.py` or `calibrate.py` — optional one-off tooling that writes debug images to disk. Used to gather templates (hoops) or visualize HSV masks (chopping).
- `assets/` — templates, captures, or calibration output.

When adding a new minigame, mirror this structure and register entry points in `pyproject.toml`'s `[project.scripts]`.

### Required conventions for any new minigame's `main.py`

- **Wrap `run()` in a `session_log` context.** The pattern is:
  ```python
  from common.session_log import session_log
  LOGS_DIR = Path(__file__).parent / "assets" / "logs"

  def run():
      with session_log(LOGS_DIR) as log_path:
          print(f"Session log: {log_path}")
          _run_inner()

  def _run_inner():
      # actual main loop
  ```
  This tees stdout to `assets/logs/session_<timestamp>.log` so a maintainer can review the bot's output without copy-paste from the terminal.

- **Load region coordinates from `assets/regions.json`, not hardcoded constants.** Pickers (`*-pick-<name>-region`) write to that file via `common.regions.save_region`; `main.py` reads via `common.regions.get_region(_HERE, "<name>")`. Hardcoded values are only acceptable as fallback defaults:
  ```python
  from common.regions import get_region
  _HERE = Path(__file__).parent
  SCORE_REGION_REL = get_region(_HERE, "score") or {"left": ..., "top": ..., ...}
  ```

- **Don't ask the user to copy-paste anything into source.** Any setup script (region picker, template cropper, calibration) must persist its result to a file the bot reads on next run.

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
