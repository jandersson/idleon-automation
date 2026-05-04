# Darts strategy notes

What the bot assumes about the game and how each piece of the throw
pipeline fits together. Living document — update as we learn more.

## What this minigame is

In-game title: **"THROWY DARTS"** (visible top-right of the playfield).
On the wiki this is the **Nine Darter** minigame — the trophy goal is
**Nine Dart Finish: 9 bullseyes in a row**, awarding +50 Defence,
+9 Weapon Power, +9% Drop Rate Multi.
[wiki](https://idleon.wiki/wiki/Nine_Dart_Finish)

Visual layout (verified from `assets/monitor/throw_*/pre_throw.png`):

- Player stands on a small wooden platform, dart held high, arm
  sweeping vertically.
- Wall behind the player is divided into vertical colored **stripes**
  (not concentric rings — the wiki summary that says "rings" is
  wrong for this game).
- Right side of the playfield has a **column of 9 pole indicators** —
  presumably the Nine Dart Finish progress / consecutive-bullseye
  counter, but unverified.
- Top-left readout shows current "Throws" / "Now" totals; bot's
  `score` region targets the "Now" digits (the running point total).
- Top-right has a "Current Reward" number that ticks down over time
  or per non-bullseye — distinct from the score we read; not
  currently used by the bot.

Wind indicator (from `assets/wind_samples/`): a small blue arrow on a
wooden panel, with multiple discrete directional states observed
(up-left, up-right, down-right). Some samples are blank wood — wind
either disabled in those moments or the region drifts during scene
transitions.

## Open questions (in-game verification needed)

These can be answered by playing a session and watching, faster than
trawling wiki/Reddit:

- **Per-stripe scoring**: how many points each colored stripe gives,
  and whether the centre stripe is the "bullseye" the trophy counts.
- **Wind effect magnitude**: does each discrete wind state offset the
  dart's flight by a fixed pixel amount, and in which direction
  relative to the arrow?
- **Wind state cycle**: deterministic (every N throws) or random; how
  often it changes.
- **Sweep-speed escalation**: wiki says "sweep speed increases over
  time" — within a single session, or only across milestone scores?

## Game-physics assumptions (empirical)

- **Throw is fixed-power; aim is determined by the player's arm angle
  at click time.** The arm sweeps continuously through release angles;
  clicking freezes the arm and launches the dart along the angle it was
  passing through.
- **Wind shifts the dart's flight in some direction.** Visualised by
  the wind indicator in the top-right of the playfield. The wind state
  changes between throws. Currently we capture wind crops per throw but
  don't yet use them to correct aim.
- **Click position has no measurable effect on aim** beyond timing.
  Same principle as hoops; bot clicks in the middle of the window for
  convenience.

## How a throw works (end to end)

1. Watch for the player's arm to sweep into the captured release pose
   (template match, conf ≥ `RELEASE_THRESHOLD`).
2. Click *immediately* — `pyautogui.click` at window centre. Latency
   between pose detection and click landing drifts the arm a few
   degrees off the captured angle. Bookkeeping (score capture, wind
   crop/diff/save, monitor folder allocation) runs after the click.
3. Capture flight frames every `FLIGHT_POLL=50ms` during the cooldown
   so a later trajectory module can extract dart-landing position.
4. Read the score region post-cooldown; pixel-diff against the pre-throw
   crop tells us hit/miss.
5. Save pre/post screenshots, wind crop, flight frames, and `meta.txt`
   to a per-throw folder under `assets/monitor/`.

## Score detection

`common.score_diff.score_changed` runs Otsu binarization on the score
crop before/after each throw and flags a change if mean abs-diff > 3.
Spot-check across the last 30 throws (2026-05-05): 10 hits (diff 6–32,
clearly separated), 16 clean zero-diff misses, 2 borderline (diff
2.25 / 2.63 — just under threshold). Detector looks roughly fine but
the threshold has only ~1px of headroom against the borderline cases.

Limitations:

- **No score magnitude.** We only know "score changed", not by how
  much. Darts scoring is +1/+2/+3 depending on which ring; we'd need
  OCR or template-matched digits to log the actual increment.
- **Threshold could be ambiguous.** Two throws sat at diff=2.25 / 2.63
  — without ground truth we can't say whether those were real hits or
  background noise. OCR (next item) would resolve this.

## Open problems

1. **Wind isn't yet a feature in any predictor.** Wind crops are saved
   per-throw but no model consumes them. The future release-angle
   predictor will need to encode wind state somehow (cluster by visual
   similarity, OCR the indicator if it's numeric, or feed the raw crop
   to a small CNN).
2. **No release-angle predictor at all yet.** Bot fires on any frame
   where the captured release pose template matches above threshold —
   meaning it always throws at the same arm angle, and corrections
   come only via swapping the captured release pose template. This is
   the eventual ML target.
3. **Game-over detection is heuristic.** Currently "no release pose
   matched in `GAME_OVER_NO_POSE_SEC=25s`" — fires when the dartboard
   scene is replaced. False fires possible during slow cycles; false
   negatives if the scene happens to keep matching.

## Aim-quality backlog

Open ideas for raising the per-session hit rate, ordered by expected
value-per-effort. Drop entries as they ship.

- **Trajectory module: dart-landing-x extraction.** Flight frames are
  now recorded (2026-05-05). Next step is a `common.dart_trajectory`
  helper analogous to `common.ball_trajectory` that runs over a throw
  folder and extracts where the dart visually lands relative to the
  bullseye. Without this the recorded frames are dead weight.
- **Score OCR / template-matched digits.** Replaces the binary "did it
  change" signal with the actual score number. Tells us +1 / +2 / +3
  per throw, which is the label any release-angle predictor would
  train against. Same path as the hoops `metrics_backlog.md` "Higher
  cost" item — pixel-art digits favor template matching over
  Tesseract.
- **shot_log SQLite parity with hoops.** Darts currently logs only to
  meta.txt and console. A `darts.db` with one row per throw (release
  pose, conf, wind hash, score_increment, dart_landing_x once
  extracted, code_commit, session_started) would let the same kind of
  per-shot queries we run for hoops, and is the prerequisite for
  fitting any predictor.
- **Wind-conditioned release angle.** Once score increment + dart
  landing offset + wind state are all logged, fit a predictor like
  hoops' GP that maps `(wind_state) → optimal_release_pose_offset`.
  Existing release pose template gives the centre; predictor outputs
  a delta. Unlike hoops there's no continuous knob to turn — release
  pose is a discrete frame match — so this likely takes the form of
  picking *which* of several pre-captured release-pose templates to
  match against given the current wind.
- **Click-timing audit on every minigame** — done for hoops and darts
  (2026-05-04 / 05). When adding a new minigame, audit the path from
  trigger detection to click landing for any disk writes or full-
  window grabs sneaking in. See CLAUDE.md "Click timing".
- **Better game-over detection.** Match the game-over screen template
  rather than relying on a no-pose timeout. Cuts false fires and false
  negatives both. Same approach as `find_game_over` in hoops.
