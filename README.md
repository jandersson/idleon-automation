# idleon-automation

> ⚠️ **Work in progress.** Nothing here is finished. Bots half-work, configs are in flux, expect breakage. See the Status table below.

[Legends of Idleon](https://www.legendsofidleon.com/) is a free 2D idle MMO with a pile of click-the-right-moment minigames gating its rewards. This repo automates them.

An experiment: **how much of a working game-bot suite can I get just by prompting Claude Code?** Every line of code in this repo was written by Claude. My role is the eyeballs and hands — running the bot, describing what's happening on screen, cropping templates in Photoshop, telling Claude which direction the shots miss. Claude does the rest: reading the game, picking detection strategies, tuning offsets, breaking deadlocks, scaffolding new minigames.

Below: bots for Idleon minigames. Each one captures a region of the screen, detects game state with OpenCV, and fires synthetic clicks via `pyautogui`. The Status table tracks how far the experiment has gotten on each.

## Status

| Module   | Working | Notes |
|----------|---------|-------|
| hoops    | ~40%    | Rim-only template matching (multi-scale, fraction-based regions). Make detection requires multi-pass Tesseract OCR + ball-trajectory cross-check — false positives are rare. Predictor is swappable (`PREDICTOR_KIND` in `main.py`): KNN over past makes (default) or bivariate OLS. Shots logged to `assets/shots.db` with full diagnostics; predictor retrains from past makes on every session start. Per-shot perturbation sweep on consecutive misses at the same hoop. Auto-snapshot + auto-commit at session end. Score 10+ (moving platform) and 20+ (moving hoop) not yet handled — need re-enabling `home_x` machinery and per-shot hoop re-detection respectively. See `minigames/hoops/STRATEGY.md`. |
| darts    | ~30%    | Release-pose template matching fires consistently. Wind region is captured, dedup-saving distinct states builds the wind library as you play. Score-region diff per throw. Multi-template per spawn-height (the dominant accuracy variable) not yet built. |
| chopping | ~70%    | Working end-to-end: position-aware eased time-to-red fire gate (`detector.eased_time_to_red_ms`; kill boundary back-tested at 110–116ms vs a 120ms budget), post-chop fire hold keyed on the zone re-roll ack, same-sweep gold upgrade, exit-on-starve banking. Every click + full-rate polls logged to `assets/chopping.db`; live PTS OCR needs a one-time `chopping-pick-score-region`. Mechanics model in `docs/chopping_notes.md`; open threads (hyperspeed ramp, gold-spawn triggers, cross-pass gold waiting) tracked in [#46](https://github.com/jandersson/idleon-automation/issues/46). |
| catching | Scaffold| Folder + entry points (`catching`, `catching-capture`, `catching-pick-play-region`). Flappy-Bird-style: click for altitude, navigate hoops. Need fly + hoop-gap detectors before this runs — currently the detectors return None. |
| mining   | ~30%    | Full detection pipeline live: plank auto-anchor, background-masked cart matching (tracks through the whole jump arc), pit + ore scanners, plank-relative PTS OCR (digits 0-4), Play-Game template match. Policy: jump-to-clear pits (validated live), jump+slam on ore (jump validated; slam window retuned after one early-slam death, awaiting a live run), airborne guard, trigger windows scaled live by a scroll-velocity estimate (the track accelerates ~40% within a run). Every click logged to `assets/mining.db` with detector state + outcome; `mining --watch` observe mode logs human play for calibration. Open work under [`label:minigame:mining`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+label%3A%22minigame%3Amining%22) and `docs/mining_plan.md`. |

## Supported platforms

The bot is a desktop screen-reader; it works on platforms where Idleon runs in a window the OS exposes via standard APIs.

| Platform | Status         | Notes |
|----------|----------------|-------|
| Windows  | ✅ Supported   | Developed on Windows 11. All scripts tested here. |
| macOS    | ❌ Untested    | `pygetwindow` (the dep used to find the Idleon window) doesn't support macOS. Would need a Quartz/AppKit-based shim. |
| Linux    | ❌ Untested    | Same — `pygetwindow` is Windows-only. An xdotool / wmctrl wrapper could substitute. |
| iOS / Android | 🚫 Out of scope | Different process model, no desktop window to read. Would need to be rebuilt on top of a mobile automation framework (Appium etc.). |
| Web      | 🚫 Out of scope | Browser-sandboxed; would need a browser-extension or a CDP/Playwright-driven approach instead of OS-level capture and click. |

Idleon runs on Steam (Win/Mac/Linux), iOS, Android, and the web; this bot specifically targets the desktop Steam window. A Mac/Linux port is feasible — only `common/window.py` is platform-specific. PRs welcome.

## Setup

Tested on Python 3.11+ on Windows.

1. Install [Python 3.11+](https://www.python.org/downloads/).
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) — Astral's fast Python package manager. On Windows:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. Clone and install dependencies:
   ```bash
   git clone https://github.com/jandersson/idleon-automation.git
   cd idleon-automation
   uv sync
   ```
4. Open Idleon, navigate to the minigame you want to bot, then in the repo directory run e.g. `uv run hoops`. The bot waits 2 seconds before clicking — switch focus to the game window in that time.

(Alternative without `uv`: `pip install -e .`, then call commands directly: `hoops`, `darts`, etc.)

## Running the tests

```bash
uv run pytest
```

Small pytest suite under `tests/` covering the pure-logic helpers:
fractional region round-trips, multi-scale template matching against
synthetic images, chopping zone lookup, score-diff binarization, and
hoops offset interpolation. Doesn't touch the screen or the game.

CI runs the suite on every push to `main` and every pull request via
[GitHub Actions on `windows-latest`](.github/workflows/tests.yml).

## Aborting a bot

Two ways to stop a running bot:
- **`Ctrl-C`** in the terminal — the normal interrupt.
- **Slam the mouse into any screen corner** — pyautogui's built-in failsafe. Useful when the terminal isn't focused.

## Minigames

### Conventions across all minigames

- **Region coordinates live in `minigames/<game>/assets/regions.json`** as fractions of the window size, so a window resize doesn't break them. You don't edit this file directly — `*-pick-*-region` scripts open a popup, you click two corners around the target, and the script writes the fractions for you. See the strategy notes in `minigames/hoops/STRATEGY.md` for the bottom-of-rim-aim rationale and other deep details.
- **Per-session logs** under `minigames/<game>/assets/logs/` capture the bot's full stdout for review.
- **Per-shot monitor folders** (where applicable) under `minigames/<game>/assets/monitor/` save pre/post screenshots, mid-flight frames, and metadata for offline tuning.

### 🏀 hoops

Basketball minigame. Platform oscillates vertically (and horizontally at score ≥10); hoop repositions randomly between shots (and moves continuously at score ≥20). Bot detects both, fires a click when the platform crosses a target Y derived from `OFFSET_ANCHORS`. Resolution-agnostic via multi-scale `matchTemplate`. Game-start prompt detection skips counting the wake-up click; game-over detection exits cleanly with final session stats.

**First-run setup:**

```bash
# Capture frames during a live game
uv run hoops-capture
# In assets/captures/, crop tight templates and save as:
#   minigames/hoops/assets/hoop.png       (rim + backboard)
#   minigames/hoops/assets/platform.png   (the wooden platform only)
# Pick the score-region (the digits, no "Score:" label):
uv run hoops-pick-score-region
# After your first game-over screen appears, capture its template:
uv run hoops-pick-game-over
```

**Then run:** `uv run hoops`

**Tuning knobs** in `minigames/hoops/main.py`:
- `SHOT_STRATEGY` — `"direct"` (default; tune offsets so the ball arc passes through the rim) or `"overshoot"` (aim past the rim and rely on `_try_rescue` clicking the ball mid-flight to drop it through). Each has its own `OFFSET_ANCHORS` and `BALL_X_TOLERANCE`.
- `OFFSET_ANCHORS_DIRECT` / `OFFSET_ANCHORS_OVERSHOOT` — list of `(hoop_y, offset)` anchor points; offset for any hoop_y is linearly interpolated between them. Higher offset = fire when platform is lower = more arc.
- `Y_TOLERANCE`, `REQUIRED_DIRECTION`, `POST_SHOT_COOLDOWN` — firing-window controls.
- `X_TOLERANCE` — at score ≥10 the platform moves horizontally; the bot anchors a `home_x` and only fires when within tolerance. Set huge (default `9999`) to disable.
- `RESCUE_WINDOW`, `BALL_X_TOLERANCE` — mid-flight ball drop. Ball detection uses motion masking (frame-to-frame diff ANDed with the orange HSV mask) so it doesn't lock onto the static rim.
- `MONITOR_MODE` — set False to skip per-shot screenshot dumps.

**Helper scripts:**
- `hoops-capture` — burst-capture for templates.
- `hoops-debug` — annotates one frame with the best hoop+platform matches.
- `hoops-ball-calibrate` — fire one shot, burst-capture the flight, overlay HSV mask for ball tuning.
- `hoops-score-calibrate` — save the score-region crop for visual verification.
- `hoops-pick-score-region`, `hoops-pick-game-over` — one-time region/template picks.

### 🎯 darts

Throwy Darts. Player teleports to a new platform position per throw; arm sweeps in a `)` arc; wind affects trajectory. Bot template-matches a `release.png` (player+arm at the captured release angle); when matchTemplate confidence peaks, fires a click. Score region diff and a wind-state library accumulate as you play.

**First-run setup:**

```bash
uv run darts-pick-score-region
uv run darts-pick-wind-region
uv run darts-capture          # burst-capture frames showing the arm at various angles
uv run darts-auto-crop-release # default frame 8 (slightly elevated arm)
```

**Then run:** `uv run darts`

Each throw saves a folder under `minigames/darts/assets/monitor/` with pre/post screenshots, the wind-region snapshot, and metadata.

**Helper scripts:** `darts-capture`, `darts-pick-release` (manual click-corners crop), `darts-auto-crop-release` (motion-detection crop), `darts-watch-wind` (standalone wind-sample collector).

### 🪓 chopping

Click "Chop" when the sliding leaf is over the green (1pt) or gold (2pt) zone. The leaf scrolls in a strip *above* the colored bar, so the bot uses two regions: a `leaf` strip (for X position) and a `bar` strip (for zone color at that X). Per community wisdom, the hitbox is the **left edge of the leaf**, not its center.

**First-run setup:**

```bash
uv run chopping-pick-leaf-region   # the leaf's track above the bar
uv run chopping-pick-bar-region    # the colored zone strip
uv run chopping-pick-button-region # the "Chop" button
uv run chopping-calibrate          # dumps masks/overlays to assets/calibration/
```

**Then run:** `uv run chopping`

Tune `LEAF_HSV` / `GREEN_HSV` / `GOLD_HSV` / `RED_HSV_LOW` / `RED_HSV_HIGH` in `minigames/chopping/detector.py` if calibration shows a mask catching the wrong things.

### 🪰 catching

Flappy-Bird-style: click to gain altitude, navigate the fly through a series of hoops. **Currently scaffold only** — needs a fly template and a working `find_next_gap` implementation before it runs.

**Setup so far:** `catching-pick-play-region`, `catching-capture`. Real detector code is the next step.

### ⛏️ mining

Cart-jump-slam minigame, rendered as a small overlay panel anchored above the player's head. Click to jump, click again mid-air to slam down. Land slams on ore deposits (the rebound is a free jump that chains into the next slam), avoid pits.

**Run it:** `uv run mining` (auto: clicks Play Game, jumps pits, jump-slams ore) or `uv run mining --watch` (observe-only: logs your clicks + detector state to `mining.db` and captures PTS digit crops — spend a scarce daily attempt on calibration data instead of an untuned bot). Add `--save-frames` to dump every frame for offline analysis. Tuning knobs live at the top of `minigames/mining/main.py`; trigger windows are distances at a reference scroll speed, scaled live as the track accelerates (`policy.ScrollVelocity`). Open work is tracked as GitHub issues under [`label:minigame:mining`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+label%3A%22minigame%3Amining%22), with the working plan in `docs/mining_plan.md`.

> Daily attempts are pooled across the in-map minigames (chopping, catching, mining, etc.). Hoops and darts use their own permanent items on individual cooldowns and don't draw from the same pool.

## Layout

```
common/
  capture.py       mss-based screen grab
  input.py         pyautogui clicks with jitter + delay
  window.py        find game window by title substring
  region_picker.py click-two-corners popup helper
  regions.py       load/save per-minigame regions.json
  templates.py     multi-scale matchTemplate helpers
  session_log.py   tee stdout to per-session log files
  shot_log.py      SQLite schema + log_shot + fetch_makes
  predictor.py     Predictor protocol, KnnPredictor, BivariatePredictor
  ball_trajectory  HSV+motion ball tracker over flight frames
  score_ocr.py     multi-pass tesseract score reader
  idleon_save.py   read & decode the local Idleon save (Haxe format)
  auto_commit.py   stage+commit+push a single file from inside a bot
  review_nag.py    "you've played N sessions since last review" nudge
  tries_counter.py persisted shared chopping/catching/mining tries count
minigames/
  hoops/           basketball — see minigames/hoops/STRATEGY.md
  darts/           throwy darts — release-pose template, wind capture
  chopping/        leaf+zones HSV bot, separate leaf/bar regions
  catching/        scaffold — flappy-bird-style, detectors not built
  mining/          scaffold — cart-jump-slam, anchor-template work pending
ui/
  launcher.py      Tk launcher: Bots / Setup / Frames tabs
scripts/
  dump_shots.py    snapshot shots.db → tracked JSON
  dump_idleon_save inspect the decoded Idleon save
docs/
  cleanup_backlog  historical record (open items → issues label:area:cleanup)
  metrics_backlog  historical record (open items → issues label:area:metrics)
  mining_backlog   reference frames + mechanics (open items → issues label:minigame:mining)
```

## Where to start (coming back to this after a while)

1. **`README.md`** (this file) — what works, what doesn't, how to install.
2. **`CLAUDE.md`** — architecture conventions + the "Hoops: established
   findings" section (settled experiments not to redo).
3. **`minigames/hoops/STRATEGY.md`** — end-to-end pipeline for the
   hoops bot: detect → predict → fire → validate → log → snapshot.
4. **GitHub Issues** — open work, labelled by area / minigame:
   <https://github.com/jandersson/idleon-automation/issues>.
5. **`TODO.md`** — cross-cutting open work + scheduled-agent IDs that
   don't fit a single issue.
6. **`docs/`** — longer-form notes and historical backlog records.
   Browseable on GitHub Pages:
   <https://jandersson.github.io/idleon-automation/>. Highlights:
   - **`docs/predictors.md`** — refresher notes on KNN, OLS, and the
     planned GP regression upgrade for the hoops predictor (renders
     with proper LaTeX on Pages).
   - **`docs/predictor_playground.html`** — interactive heatmap
     visualisation of the predictors over a synthetic playfield.
   - **`docs/cleanup_backlog.md`** + **`docs/metrics_backlog.md`** +
     **`docs/mining_backlog.md`** — historical record of settled
     conclusions and reference context. Open follow-ups live in
     issues now.

`uv run pytest` should always pass — that's the executable spec.
