# idleon-automation

> ⚠️ **Work in progress.** Nothing here is finished. Bots half-work, configs are in flux, expect breakage. See the Status table below.

An experiment: **how much of a working game-bot suite can I get just by prompting Claude Code?** Every line of code in this repo was written by Claude. My role is the eyeballs and hands — running the bot, describing what's happening on screen, cropping templates in Photoshop, telling Claude which direction the shots miss. Claude does the rest: reading the game, picking detection strategies, tuning offsets, breaking deadlocks, scaffolding new minigames.

Below: bots for Idleon minigames. Each one captures a region of the screen, detects game state with OpenCV, and fires synthetic clicks via `pyautogui`. The Status table tracks how far the experiment has gotten on each.

## Status

| Module   | Working | Notes |
|----------|---------|-------|
| hoops    | ~60%    | Reliable score 0-9 (~7/10 makes). Force-fires unreachable hoops to break deadlocks. Mid-flight ball-drop rescue scaffolded, HSV not yet calibrated. Score 10+ horizontal-platform handling fragile; score 20+ (moving hoop) not handled. |
| chopping | Untested| Code in place; user hasn't calibrated HSV ranges or run yet. |
| darts    | Scaffold| Folder + entry points (`darts`, `darts-capture`) exist. Mechanics unknown; awaiting capture frames before writing the detector. |

## Setup

Requires Python 3.11+.

```bash
pip install -e .
```

## Aborting a bot

Slam the mouse into any screen corner — pyautogui's default failsafe kills the running bot. Useful when it's spam-clicking and you've lost the keyboard.

## Minigames

### hoops

Basketball minigame — the character stands on a platform that oscillates vertically, and the hoop randomly repositions on the right side after each shot. The bot detects both, and left-clicks when the platform crosses a target height (derived from the hoop's rim).

**Game mechanics** (per [IdleOn wiki](https://idleon.wiki/wiki/Hoops_Minigame)):
- Difficulty ramps with score *within a single trial*:
  - Score ≥10: the player's platform starts moving **horizontally** in addition to vertically.
  - Score ≥20: the hoop also starts moving horizontally.
  - Score ≥30: the hoop moves both horizontally and vertically.
- Clicking the **ball mid-flight** makes it drop straight down — a manual rescue for shots overshooting the rim. The bot doesn't use this yet.
- Scored across 3 trials; combined score of 66 unlocks the Hoops pet.

The current bot only handles the score-0–9 regime (vertical platform, static hoop position between shots). Beyond score 10 it will start missing — the platform's horizontal motion isn't tracked, and beyond score 20 the hoop position needs re-detection per frame, not just after each shot.

Before running:
- Drop two cropped screenshots in `minigames/hoops/assets/`:
  - `hoop.png` — just the rim
  - `platform.png` — the platform the character stands on
- If the game's window title isn't "Idleon", update `WINDOW_TITLE` in `minigames/hoops/main.py`

Grab frames for cropping via burst capture (30 frames over ~3s through the same capture path the bot uses, so templates are pixel-identical to what gets matched):

```bash
uv run hoops-capture
```

Frames land in `minigames/hoops/assets/captures/`. Crop `hoop.png` and `platform.png` from whichever frames show each most clearly.

**Tuning knobs** in `minigames/hoops/main.py`:
- `OFFSET_ANCHORS` — list of `(hoop_y, offset)` anchor points; the launch offset for any hoop position is linearly interpolated between them. Add a new anchor when shots consistently miss in a particular hoop-Y range. Positive offset fires later (platform lower); negative fires earlier (platform higher).
- `Y_TOLERANCE` — how close platform Y has to be to target before firing.
- `REQUIRED_DIRECTION` — `"up"`, `"down"`, or `"any"`. Restricts firing to the rising or falling half of the oscillation.
- `POST_SHOT_COOLDOWN` — seconds to wait after firing before re-detecting the hoop.
- `X_TOLERANCE` — at score ≥10 the platform also moves horizontally; the bot anchors a `home_x` from early frames and only fires when within this tolerance. Set huge (e.g. `9999`) to disable.
- `RESCUE_WINDOW`, `BALL_X_TOLERANCE` — controls the mid-flight ball-drop rescue. After the launch click, the bot tracks the ball for `RESCUE_WINDOW` seconds and clicks it (drops it straight down) when it's within `BALL_X_TOLERANCE` of the hoop X.
- `SCORE_REGION_REL` — window-relative crop of the score readout. Diffed before/after each shot to detect makes. Set to `None` to disable shot-result logging. Verify the crop with `uv run hoops-score-calibrate`.

Helper scripts (all registered as `[project.scripts]`):
- `hoops-capture` — burst-capture frames for cropping `hoop.png` / `platform.png` templates.
- `hoops-debug` — annotate one frame with the best hoop and platform matches and their confidences.
- `hoops-ball-calibrate` — fire a shot, burst-capture the flight, overlay the HSV mask so you can tune `BALL_HSV_LOWER`/`UPPER` in `detector.py`.
- `hoops-score-calibrate` — save the score-region crop for visual verification.

```bash
hoops
```

### chopping

Clicks the chop button when a sliding pointer is over the green (safe) or gold (bonus) zone. Color-based, not template-based.

Before running:
- Set `BAR_REGION_REL` and `CHOP_BUTTON_REL` in `minigames/chopping/main.py` — both are **relative to the game window's top-left**, not the full screen
- Tune HSV ranges in `minigames/chopping/detector.py` if the defaults don't match
- If the game's window title isn't "Idleon", update `WINDOW_TITLE`

Calibrate first — dumps masks and overlays to `minigames/chopping/calibration/` so you can see which pixels each HSV range is catching:

```bash
chopping-calibrate
```

Then run:

```bash
chopping
```

## Layout

```
common/          shared screen-capture + input helpers
minigames/
  hoops/         template-matching bot + ball-rescue + score diff
  chopping/      HSV-zone bot + calibration script
  darts/         scaffold only — mechanics not yet implemented
```
