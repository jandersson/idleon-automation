# idleon-automation

Screen-reading bots for Idleon minigames. Captures a region of the screen, detects the game state with OpenCV, and fires synthetic clicks via `pyautogui`.

## Setup

Requires Python 3.11+.

```bash
pip install -e .
```

Failsafe: slam the mouse into a screen corner to abort any running bot (pyautogui default).

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
- `VERTICAL_OFFSET` — pixels added to the rim Y to get the ideal platform launch Y. Rolls rim-vs-platform geometry *and* ball-travel lead into one number. Start at 0, adjust based on whether shots land short or long.
- `Y_TOLERANCE` — how close platform has to be to target before firing.
- `REQUIRED_DIRECTION` — `"up"`, `"down"`, or `"any"`. Whether to only shoot on the rising half of the oscillation, falling half, or both.
- `POST_SHOT_COOLDOWN` — seconds to wait after firing before re-detecting the hoop.

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
  hoops/         template-matching bot
  chopping/      HSV-zone bot + calibration script
```
