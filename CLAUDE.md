# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Screen-reading bots for the game **Idleon**. Each bot grabs a region of the screen, runs OpenCV against it, and fires synthetic clicks via `pyautogui`. Windows-only in practice (depends on `pygetwindow`).

The README covers per-minigame run commands and tuning knobs — don't duplicate that here. This file documents the cross-cutting patterns.

## Workflow

**Always commit changes.** After completing a logical unit of work, commit it without waiting to be asked. Split unrelated changes into separate commits. Pushing follows the daytime-hours rule from user memory; committing has no time gate.

**Commit message format: Conventional Commits.** `<type>(<scope>): <description>`. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`. Scope is the affected area (`hoops`, `darts`, `launcher`, `common`, etc.). Breaking changes use `!` after the type/scope (`feat(hoops)!:`). Body and footer are optional; keep the subject line under ~72 chars and use the body for the *why*. The legacy log has many `scope:` (no type) commits — don't follow that pattern, use the full Conventional format for new work.

**Push directly to `main`.** Solo repo; the per-change PR roundtrip isn't worth the overhead. Was tried briefly on 2026-05-24 (#29, #30) and reverted — branches added latency without value. The Claude PR review bot (#28) is still wired up for any PRs that do get opened (e.g. external contributions), but the default flow is commit + push to main.

**Concurrent CLAUDE CODE sessions → one git worktree + branch each.** The exception to "push to main": when two **Claude Code dev sessions** edit the repo at the same time — e.g. one improving darts (in automode), one on fishing. This is about the *coding sessions*, not the game-playing bots. Both sessions in the same directory share one working tree, **index**, and `HEAD`, so one session's `git commit` sweeps the other's staged files into its commit (observed 2026-06-18: staged fishing changes landed in the other session's `fix(darts)` commit). Branching alone doesn't help — a single checkout has one index. The fix is isolation *before* launch:

```bash
git worktree add ../idleon-<scope> -b <scope>     # e.g. ../idleon-darts -b darts
git -C ../idleon-<scope> push -u origin <scope>   # set upstream (push.default=simple needs it)
```

Launch the second session with that dir as its cwd (a session's cwd is fixed at start, so this must precede launch). Each session commits to its own branch.

**Merge to `main` yourself when finished** (the branch is just a staging lane, not a fork — restore trunk). Do it FROM your branch; never `git checkout main` (two worktrees can't both hold `main`, and concurrent finishes would race):

```bash
git fetch origin
git merge --no-edit origin/main      # fold in the other session's work
uv run pytest -q                     # don't merge red
git push origin HEAD:main            # fast-forward main
```

This is **race-safe**: if the other session pushed `main` first, `push HEAD:main` is rejected as non-fast-forward — re-`fetch`, re-`merge origin/main`, retry (nothing is lost; the work just waits on the branch). An autonomous session MUST do that retry, or at least flag the rejected push rather than assume `main` updated. Merges auto-resolve only for **disjoint** dirs (darts vs fishing never touch the same files); on a real conflict in a **shared** file (`common/`, `pyproject.toml`, `CLAUDE.md`, shared docs) **bail — leave it on the branch and flag it**, don't force a resolution. Then `git worktree remove ../idleon-<scope>` (and delete the merged branch). (Aside: a game bot run from a worktree auto-commits its DB to that worktree's branch, so `darts.db`/`fishing.db` stay on-branch — no extra handling.)

**Add unit tests for new stuff.** New pure-logic helpers, parsers, schema modules, regression guards — write a test alongside the change in `tests/`. Match the existing style: fast, self-contained, no live captures. CV-against-real-frames stays manual (visual, calibrated by user).

**File an issue for every refinement you find.** A refinement, deferred edge case, approximation a better approach could replace, follow-up a fix unblocks, or "looks off" anomaly that you notice but don't finish now goes in a GitHub issue (`gh issue create`, matching the existing label scheme) — not just chat, a code comment, or a docs caveat. Bias to over-file; cross-link the issue with the docs/code. See [`.claude/AGENTS.md`](.claude/AGENTS.md).

## Setup

`pip install -e .` or `uv sync` — both work, `uv.lock` is checked in. Python 3.11+.

**Optional: Tesseract OCR** for score-int logging in shots.db. Without it, OCR'd score columns are NULL but everything else works. Install on Windows:

```
winget install --id=UB-Mannheim.TesseractOCR
```

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

- **Minigame "Play Game" buttons are dynamic, not static.** In Idleon the entry prompt for each minigame (chopping/catching/mining/etc.) renders as a UI overlay anchored above the player character's world position — so the button's screen coords change when the player moves, the window resizes, or the user switches between environments. Saving a fixed regions.json rectangle works for one capture and breaks on the next. Instead, **template-match the button visually** via `common.templates.match_multiscale_center` against an extracted sprite under `assets/play_button.png` (or per-resolution variants). The minigame cart, start prompts, and overlay panels in general behave the same way — assume any in-world UI element needs visual detection, not coordinate caching. See `minigames/mining/detector.py:find_play_button` and `find_cart` for the canonical pattern.

- **Log every decision to a per-bot SQLite DB at `assets/<bot>.db`.** One row per click / shot / throw / jump captures the detector state at fire time + the outcome (measured a beat later). The shape is bot-specific — hoops logs `hoop_x, platform_y, offset, made` etc; darts logs `release_pose_x, launch_angle, hit`; mining logs `cart_x, next_distance_px, outcome`. Don't try to share a schema across bots, but DO follow the same data-tracking pattern: every fired action goes in the DB, with code_commit / session_started / source columns common across bots. Querying answers questions like "at what trigger distance does the bot actually survive?" that grepping log files can't. See `common/shot_log.py` (hoops), `minigames/darts/shot_log.py`, `minigames/mining/jump_log.py` for examples. New bots should mirror this layout: schema module next to `main.py`, `open_db()` + `log_*()` + outcome-update helpers, ALTER TABLE migration list for late-added columns.

  **The DBs are tracked in git and auto-committed at session end** (via
  `common.auto_commit.commit_file_if_changed` in each bot's `run()` exit
  path) so other machines get the data with `git pull` — the bots only run
  on the Windows box, so the flow is one-way and merge conflicts can't
  happen. Debug captures (`monitor/`, `captures/`, `logs/`) stay
  gitignored — they're regenerable and tens of GB. Wire the same hook into
  new bots.

  **Name the DB by the game, not the action being logged.** Darts logs to `darts.db`, mining logs to `mining.db`. The DB is a per-bot artifact and naming it after the game keeps it discoverable when scrolling `assets/` and survives the bot growing additional tables (multiple action types in one DB). Hoops' `shots.db` is a legacy exception — don't follow it for new bots. Table names within the DB should describe the action (`shots`, `throws`, `jumps`).

### Coordinate convention

All region constants in source are **window-relative**, not screen-absolute. The window's screen position is resolved at runtime via `common.window.get_bounds(WINDOW_TITLE)`, which matches by case-insensitive title substring. Capture and click coordinates are computed by adding the window's `(left, top)` each tick — so the bot survives the user moving the game window mid-run.

This also means: never hardcode screen coordinates. Always express positions relative to the window.

### Frame format gotcha

`mss.grab()` returns BGRA. OpenCV operations expect BGR. Detectors do `cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)` as their first step — preserve this when adding new detectors.

### Detection styles

Two approaches are in use; pick whichever fits the visual cue:

- **Template matching** (`hoops`): `cv2.matchTemplate` with `TM_CCOEFF_NORMED`, region-restricted (right half for hoop, left half for platform) to cut false positives. Templates must be captured *through the same pipeline as the live bot* — that's what `hoops-capture` is for. Cropping a template from a manual screenshot will mismatch because of color-space and scaling differences.
- **HSV color masking** (`chopping`): `cv2.inRange` on HSV channels, with priority resolution (gold > green > red). Always tune via the `<minigame>-calibrate` script, which dumps per-range mask overlays to `calibration/`.

### Click timing

When a bot samples a moving world state to decide *when* to fire, fire
the click immediately after the decision — no disk writes, no extra
`grab_region` calls, no `random_delay` in between. World state keeps
moving during that latency, biasing every shot away from what was
sampled. Bookkeeping (pre-shot frame saves, score/lives region captures,
randomization) goes after the click.

Concrete cases in the codebase:

- **Hoops** (settled 2026-05-04): pre-click `save_frame`, two extra
  full-window `grab_region` calls for score/lives, and a
  `random_delay(10, 40)` stacked into 50–120ms of latency — biasing
  platform_y by ~20–25px on fast-bob shots and showing up as consistent
  overshoot on perturbed retries. Score/lives "before" capture still
  works post-click because the in-game score doesn't update until the
  ball lands ~2.9s later.
- **Darts**: see the comment at `darts/main.py` above the throw `click()`
  — even 20–60ms of latency drifts the arm a few degrees off the
  captured release angle.
- **It crept back once** (hoops, caught 2026-06-10): the pre-game prompt
  check (cc42529, 2026-05-09) put a 104ms full-frame template match
  between the fire decision and the click — five days after the original
  latency fix. If a pre-click signal is computed from a frame that's
  already in memory, compute it AFTER the click; the result is identical.

### Entry points and the sys.path dance

Each `main.py` and tooling script starts with:

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

This makes `from common...` and `from minigames...` work whether launched as a module, as a script, or via the `[project.scripts]` console_scripts entry. Keep it when adding new entry-point scripts.

### Idleon clicks are buttons, not pointers

Across the minigames we've investigated, **click position has no
gameplay effect — only the click event matters.** A click anywhere in
the play area is equivalent to a "press A" on a console controller: the
game reads the timing and the in-game state, not the pixel coordinates.
This is true for hoops (verified by click-position sweep) and for mining
(jump on click 1, slam on click 2 mid-air; cart x doesn't depend on
where the click landed). Assume it holds for new minigames unless
something explicitly proves otherwise.

Implication: bots can fire `click()` at any sane position within the
game window. Using a detected sprite's coordinates (hoop center, cart
center) is just a sane default — it's not aiming. Don't try to
correlate `click_x / click_y` with outcomes in the DB.

### Hoops: established findings

A few things were settled empirically on 2026-05-03 — don't re-run these
investigations:

- **Click position has no measurable effect on aim** — see the
  cross-cutting note above. Verified by hoops with `click_sweep` (varied
  click_y across 320px) and `click_extreme` (window corners)
  experiments; both produced ball trajectories within the ~20px noise
  floor. See `docs/cleanup_backlog.md` for the deletion log.

- **Score detection requires both OCR + trajectory cross-check.** Tesseract
  occasionally misreads small score digits (`0` → `1`) producing false-
  positive makes. The current `_log_shot_result` rejects any "OCR-said-make"
  shot where `ball_x_at_rim_height` is more than `TRAJECTORY_MAKE_TOLERANCE`
  (60px) from `hoop_x`. Don't drop the trajectory check just because OCR
  multi-pass voting is on — they're complementary.

- **Pre-shot OCR fails ~20-30% of the time** (animation residuals between
  shots). Make detection anchors on `stats["session_score"]` (a running
  high-water mark of post-shot OCR reads) instead of requiring per-shot
  pre/post agreement. Confident make count and live session score are
  reported as separate numbers.

- **Click fires before bookkeeping** — see the cross-cutting "Click
  timing" subsection above. Hoops was the case that surfaced the rule
  (2026-05-04); same principle applies wherever a sampled world state
  drives click timing.

- **`trajectory_gp` is not better than `gp`** (settled 2026-05-24,
  issue #23). An early 3-session window had `trajectory_gp` at 45%
  vs `gp`'s 23%, but after 9 more sessions (n=103, 7 added 2026-05-23/24)
  the combined rate landed at 26.2% — statistically tied with `gp`'s
  25.5% (Wilson 95% CIs overlap heavily: [18.7, 35.5] vs [15.5, 38.9]).
  The early window was small-N luck. `gp` was the default until
  `make_prob` superseded it (see next finding); `trajectory_gp` remains
  selectable via the launcher dropdown but shouldn't be promoted without
  a much larger sample showing separation.

- **`make_prob` is the validated default predictor** (`PREDICTOR_KIND`
  in `minigames/hoops/main.py`, promoted per issue #38). It models
  P(make | hoop_x, hoop_y, platform_y, platform_vy) directly — a GP
  classifier on every velocity-instrumented shot (label = respawn-
  corrected `made`, so clank=0; reach-censoring that defeats the
  trajectory regressors doesn't apply). At fire time it scores candidate
  (platform_y, vy, direction) states built from the live bob buffer and
  picks the highest-probability one, subsuming the hand-drawn direction
  thresholds (it learns where up beats down per hoop). Promotion was
  earned on the #23 discipline: model picks (`target_source='model'`)
  reached n=112, 58.9% makes, Wilson 95% [49.7, 67.6] — disjoint from
  the policy stack's gate verdict [27, 43]. Don't revert the default
  without a comparably-powered counter-sample.

- **The June 2026 miss investigation (#37) — see
  `docs/hoops_findings.md` for the full write-up.** Headlines, settled
  2026-06-09/10: misses are not aim error (in-band arrivals make at
  ~64%) — they're structure clanks misread as "way short" by
  `ball_x_at_rim_height` (use the bounce-aware arrival via
  `ball_peak_x`) plus velocity-driven over/undershoots (the ball
  inherits the platform's vertical velocity; `platform_vy` is logged
  per shot). Firing direction is a per-hoop policy
  (`_required_direction_for`): dir=down only at very low hoops (y≥530,
  dir=up proven futile by a full-bob-range exploration sweep). The
  near-clank-band (x≤640) dir=down override was tried and reverted
  2026-06-10, so above the 530 threshold the make_prob model picks
  direction per shot (its `best_dir`, not the hand rule). One predictor
  is fitted per direction.
  Make detection is multi-signal — post-shot OCR alone loses ~10% of
  makes; the hoop-respawn signal (`made_source='respawn'`) and the
  prompt-anchored score are load-bearing, and `clean_make` is never
  bypassed (lucky bounce-ins score but don't train). Pre-game shots
  (start prompt up) cost no lives and are used for bob-range
  exploration (`target_source='explore'`). Don't re-derive these;
  don't re-enable big perturbations (capped ±32, in-band holds capped
  at 2). Follow-up model work: issue #38.

### Darts: established findings

- **The stripe outcome IS steerable — vertically, via dy-at-fire**
  (corrected 2026-06-10, same day as the first analysis, which measured
  the wrong axis: landing_x is a mid-flight horizontal proxy and is
  indeed noise, but the board's stripes are stacked VERTICALLY and the
  stripe hit is set by arc height). The chain: where in the release
  pass the click lands (centroid_dy at fire) → launch angle
  (within-spawn corr +0.39..+0.79) → apex → board height → stripe
  (corr 0.41). Empirically dy=-7 fires average stripe 3.8 with 50%
  bullseyes vs ~8% at dy<=-10: late-pass (dy near 0 from below) =
  flat arc = mid-board red/bullseye. Aim features are buildable as
  dy-band fire selection; horizontal aim remains nonexistent.
- **The swing-pass discriminator is centroid-dy, not pose or conf**
  (settled 2026-06-10, issue #26): dy ≤ 0 at fire ≈ release pass
  (~90%+ hit), dy > 0 ≈ up-swing (~0%). The release threshold adapts
  per spawn (template peaks differ by spawn); the dy-gate, not the
  threshold, vets the moment.
- **Units (since 2026-06-10): the live arm-motion signal is vy in
  px/SECOND**, cadence-invariant (`arm_motion.centroid_vy_px_s`,
  logged as `arm_centroid_vy_at_fire`). Historical dy figures in this
  file and in the dormant `arm_centroid_dy_at_fire` column are px/poll
  at the ~250ms compute-bound cadence — multiply by ~4 to compare
  (static band [-8,-5] px/poll = [-32,-20] px/s). Per-poll units would
  silently rescale with every poll-loop speedup; px/s doesn't. The
  motion mask still diffs frames ~250ms apart regardless of cadence
  (`REF_MOTION_DT_S`), so MIN_AREA's dropout behavior stays calibrated.

- **The wind-conditioned E[stripe] GP is the validated default aim**
  (issue #41, promoted 2026-06-14). It scores live (platform_y, vy)
  release passes per the parsed wind (`wind_x`/`wind_y`) and fires the
  highest-EV vy band (`minigames/darts/stripe_model.py`,
  `model_vy_band`); it's already the operative aim whenever the data
  floor is met (fits at 390 clean rows ≥ MIN_SAMPLES=150), with the
  static calm band `[-32,-20]` px/s as the fallback when the GP can't
  fire (`aim_mode` tags each: `model`/`band`/`fallback`/`explore`).
  Validation on the OCR-repaired labels: model-aim throws hit **94%
  (n=137, Wilson [89,97])** vs the **same-era** (2026-06-10) baseline
  **81% (n=221, [75,86])** — disjoint CIs. The model and baseline were
  never run concurrently (sequential by session, not a true A/B), and
  the 94-vs-81 lift is the honest one — an earlier 94-vs-70 framing
  blended in the abandoned pre-fix May regime (35%), which is a
  different bot. The wind-conditioning earns its keep where it should:
  under strong wind (≥6 mph) the model holds **94% / E[stripe] 3.06
  (n=34, [81,98])** while the un-conditioned baseline collapses to
  **55% / 1.50 (n=22))** — the cleanest evidence, all same-day.
  Honest nuance / open refinement: in dead calm the hand-tuned static
  band still edges the model on stripe value (band E[stripe] ~3.5 at
  hit 100%, n=16 vs model 2.61, n=62) but can't fire under wind — a
  calm-air special-case (defer to the band when wind≈0) is a possible
  follow-up, not a blocker. Don't revert the default without a
  comparably-powered counter-sample (#23/#38 discipline).

- **Calm-air A/B is live, data-gated (#48, 2026-06-15).** The calm-air
  follow-up above is now an experiment rather than a switch: when wind
  is calm (`wind_speed < CALM_WIND_MAX = 2` mph) and the model band is
  available, the aim alternates the static band vs the model band by
  throw-index parity (`_calm_ab_arm`), tagging fires `band_ab` /
  `model_ab` so band-in-calm accumulates under the same conditions as
  model-in-calm. Only the calm regime is touched — the windy /
  no-model paths (validated #41) are byte-for-byte unchanged. Out-of-band
  passes that exhaust the skip budget still log `fallback`, so only clean
  in-band fires carry the `_ab` tags. Don't flip the calm default off the
  thin n=16: accumulate ~50–100+ `band_ab` throws, then compare
  `band_ab` vs `model_ab` (filter `wind_speed < 2`) on hit / E[stripe] /
  bullseye with Wilson CIs — disjoint in the band's favour flips the calm
  default, overlapping closes #48. **Analysis caveat (band-width
  confound):** the static band is a fixed 12 px/s wide while the model
  band is wind-conditioned and may be narrower, so a narrower `model_ab`
  band pushes its edge vys to `fallback` and biases the surviving
  `model_ab` rows toward the central (easier) vys — confounding aim
  quality with the vy distribution each arm sampled. Before trusting the
  comparison, check the two arms' `arm_centroid_vy_at_fire` distributions
  and `fallback` rates are comparable, and stratify the stripe-value
  comparison by vy range. All of that is recoverable from the log
  (`aim_mode`, `wind_speed`, `arm_centroid_vy_at_fire`, `score_increment`).
  **Second caveat (#51 regime confound):** the throw-count gate (#51)
  re-targets only the model band late-game, so once gray/tan die (≥10/25
  hits) `model_ab` aims green/red while `band_ab` stays on the static
  band — conflating aim quality with the gate. In calm air the argmax is
  usually already green/red so the band rarely shifts, but to be safe
  stratify the calm A/B to `hits < 10` (reconstruct cumulative hits from
  the per-session `hit` column) for a gate-free comparison.

- **Throw-count stripe gate is live (#51, 2026-06-16).** Low-value
  stripes stop scoring as a game progresses (idleon.wiki *Throwy Darts*
  "Until 10th/25th hit"; confirmed in darts.db over 52 games): gray (+1)
  becomes a miss once 10 hits have been made, tan (+2) once 25. The
  threshold counts cumulative HITS, not throws — tan's last scoring
  hit-number is exactly 25 (n=207) and gray's is ≤9 (n=25), which only
  lines up on the hit axis (throw-count smears below 25 once misses are
  included). The E[stripe] model is now three surfaces
  (`RegimeStripeModel` in `stripe_model.py`): full / gray-dead /
  gray+tan-dead, selected per fire by `shot_stats["makes"]` (the live
  scoring-hit count). The dead-stripe surfaces re-score those stripes to
  0 — warm-started from the full fit's optimised kernel (~0.01s each, no
  startup cost) — so the wind-conditioned band scan walks off the doomed
  vys toward green/red on its own. Only the `model`/`model_ab` aim path
  is gated; the static band already targets the mid-board, and
  explore/fallback fire on any valid pass. The gate keys on the live
  `makes` count, which can lag true hits if score detection drops (~10%),
  triggering the gate a few hits late — there's no cleaner in-game hit
  signal. The secondary red-chain/life mechanic (3 consecutive bullseyes
  = +1 life) is NOT modelled (needs life tracking): issue #56.

### Chopping: established findings

Settled with instrumented data (2026-06-11 → 2026-07-11) — don't
re-derive; full model in `docs/chopping_notes.md`, history on #46:

- **Speed ramp is TIME-accumulated, applied at chop events** (an idle
  round explodes on its first chop — the bot auto-plays for zero
  idle). The apparent ~650 px/s "saturation" is just f(t) at normal
  round lengths. Waiting ≤45s mid-round is free.
- **Green +1 / gold +2, verified per-chop** via the PTS step function
  (`polls.pts_read`; counter updates instantly, gold's second tick
  can lag seconds). Deaths bank the points — late-round risk is cheap.
- **Never cache overlay coordinates** — the overlay anchors above the
  player and moves per map. The bar is auto-located visually; every
  other region derives from it.
- **Planned shots beat reactive gating** (the gate starves where a
  74px green crosses in ~116ms). The planner's error model is fully
  measured: 12ms base sigma × (1/sinθ)^1.5 (thin edge fires died at
  29% vs mid 8%) + 5% horizon; empirical death-risk curve (the tail
  is ~50x Gaussian at floor margins); EV ranking that reorders but
  never refuses a lone candidate. Post-mortem replays must pass
  sigma_ms=12 (the signature default 16 silently rejects real fires).
- Tesseract cannot read the score's pixel font — template OCR only.

### Safety

`pyautogui.FAILSAFE = True` is set globally in `common/input.py`. Slamming the mouse into any screen corner aborts. Every `main.run()` opens with a 2-second sleep so the user can switch to the game window before clicks start. Preserve both conventions in new bots.

`common.input.click` adds ±3px positional jitter and `random_delay` adds 80–200ms by default — don't remove the randomization, it's part of looking non-bot-like to the game.
