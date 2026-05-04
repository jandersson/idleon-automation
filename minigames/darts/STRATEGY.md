# Darts strategy notes

What the bot assumes about the game and how each piece of the throw
pipeline fits together. Living document — update as we learn more.

## Cost and cadence (player budget)

Researched 2026-05-05; hypothesis "same as hoops" was **wrong**. Darts
uses a **real-time cooldown that grows with each play that day**, not
a per-day attempt cap. Resets at the daily boundary.

- **No daily attempt limit per se** — plays are gated by a real-time
  reset timer, not a counter. Game state tracks
  `Time until Darts Minigame resets` and
  `Number of Darts Minigame played today - makes successive reset timer longer`
  (OLA entry 440 per cheat-table dumps:
  [FearLess Revolution thread p586](https://fearlessrevolution.com/viewtopic.php?t=14407&start=8775)).
- **No unlimited practice mode** — unlike hoops, once the cooldown is
  active there's no free retry. You wait. Confirmed by absence in
  every public source consulted.
- **No play cost** in tickets/energy. The cooldown is the only gate.
- **No "extra plays" upgrade for darts** confirmed; the gem-shop
  /bribe "+attempts" items reportedly apply to the gathering
  minigames (mining/chopping/fishing/catching), not darts.
  ([Steam discussion](https://steamcommunity.com/app/1476970/discussions/0/3127163422371650124/))
- **Couldn't verify** (idleon.wiki returned 403 to scraping): exact
  base cooldown in seconds, the growth function per play, whether the
  daily reset zeroes the per-day counter back to the base cooldown.

### Verified from save file (2026-05-05)

`common.idleon_save.read_darts_cooldown()` reads the live cooldown
straight out of the LevelDB save. Mapping confirmed empirically by
diffing the OLA array across a 30s wait:

- `OLA[439]` = ticks of cooldown remaining (negative = playable)
- `OLA[440]` = number of plays today (matches the cheat-table)
- `OLA[442]` = base cooldown duration for the current play tier
- All cooldown values are in **5-tick-per-second** units (so 125 ticks
  = 25 seconds), verified by watching ola[439] decrement by exactly
  150 ticks over 30 seconds of wait.

Observed after 2 plays: cooldown was 25.0 seconds. Much shorter than
expected — the "grows per play" effect probably ramps up slowly. Run
the helper after a few more plays to chart the growth curve.

**Practical implication for this bot**: very different from hoops.
Each session consumes future play time. Don't spam-test like we can
with hoops practice mode — every spin-up extends the next cooldown.
Plan around it: capture monitor data deliberately, treat each session
as expensive, save offline analysis for between cooldowns.

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

## Stripe scoring (verified from monitor screenshots)

The badge row at the top of the playfield shows **+1, +2, +3, +5** —
the stripe values from outer to inner. Bullseye is **+5**. Confirmed
2026-05-05 by reading
`assets/monitor/throw_006_051203/post_throw.png`. This means the
score-magnitude approach for distinguishing bullseyes from regular
hits is viable: a bullseye throw should leave a notably larger
post-throw score increment than a +1 / +2 / +3 hit.

## Open questions (in-game verification needed)

These can be answered by playing a session and watching, faster than
trawling wiki/Reddit:
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

## Bootstrapping order (2026-05-05)

The bot is currently in a chicken-and-egg state for any predictor
work. Hits drive variation; without hits there's no signal:

- **Bot misses → no teleport, no wind change → all `darts.db` rows
  share `(default_player_x, default_wind)` → predictor has nothing to
  fit on.**

So the priority order is forced:

1. **Get hit rate up.** Tight shaft+tip release template (shipped
   2026-05-05) was the first move — old wide template lost matches
   after the post-hit teleport, killing sessions after 1–2 throws.
   Validate with a real session: pre-fix runs were 1–2 throws then
   bail; the fix should keep sessions going for ≥10 throws.
2. **First predictor: 1D `(player_x) → optimal_release_angle`.**
   Player position varies after every successful hit. As soon as
   sessions produce 20–30 hits across varied teleport positions, fit
   a simple GP/KNN on this single feature. Wind not needed yet.
3. **Add wind as 2nd/3rd feature** when hit rate is high enough that
   wind actually changes within sessions. Until then the wind state
   is mostly the constant default and adds noise.

## Aim-quality backlog

Open ideas for raising the per-session hit rate, ordered by expected
value-per-effort. Drop entries as they ship.

- ~~**Trajectory module**~~ — shipped 2026-05-05 as
  `common.dart_trajectory`. Extracts `launch_angle_deg`, `apex_y`,
  `landing_x`, and `frames_seen` from a throw folder's flight frames
  via gray-pixel HSV mask + motion mask. Tested on synthetic frames;
  spot-checked on 3 real throws (angles -22.2°, +23.1°, -31.3°).
- ~~**Score OCR**~~ — now wired via `common.score_ocr.read_score`.
  Per-throw `score_increment` (the +1/+2/+3/+5 magnitude) is logged
  alongside the binary hit/miss diff. Template-matched digits would
  still be more reliable than tesseract; tracked under hoops'
  `metrics_backlog.md` "Higher cost" — same upgrade path applies here.
- ~~**shot_log SQLite parity with hoops**~~ — shipped 2026-05-05 as
  `minigames/darts/shot_log.py` writing to `assets/darts.db`. Schema
  has release pose, launch_angle/apex/landing, score increment, hit/
  bullseye flags, streak, code_commit. fetch_hits() returns the
  trajectory-and-outcome triples ready for predictor training.
- **Validate the new tight release template in a real session.**
  Auto-extracted shaft+tip template shipped 2026-05-05; we've tested
  it against 30 capture frames (clean separation: 0.79–1.00 at
  release angle, 0.61 elsewhere) but not yet in a live session. Run
  `uv run darts` and confirm sessions go ≥10 throws instead of
  bailing after 1–2. This is the single biggest hit-rate factor on
  the table right now.
- **1D `(player_x) → optimal_release_angle` predictor.** First
  useful predictor for darts because player position varies on every
  hit, even before wind matters. Reuses `common.predictor.GpPredictor`
  / `KnnPredictor` directly — they're 2D-input but a constant
  second arg works fine. Train against `darts.db` once we have ~20+
  hits with varied teleport positions. Doesn't need wind.
- **Continuous wind centroid features.** When wind starts changing
  within sessions (i.e. hit rate is high enough), encode wind as
  `(wind_blue_centroid_x, wind_blue_centroid_y)` — centroid of blue
  pixels in the wind crop, normalised to [-1, 1]. Plus
  `wind_blue_pixel_count` for "is wind active at all". Avoids
  enumerating discrete wind states (we don't know how many exist).
  GP interpolates naturally between observed centroids, including
  wind directions never seen before.
- **Wind-conditioned release angle.** Final form once both player_x
  and wind features are flowing. Predictor input becomes 3D:
  `(player_x, wind_centroid_x, wind_centroid_y) → optimal_release_angle`.
  Release pose is template-discrete rather than a continuous knob,
  so consuming the predicted angle takes one of:
    - (a) Multiple pre-captured templates at different angles; pick
      the closest match to predicted angle and fire on that template.
    - (b) Capture a single template at the predicted angle, swap it
      in, fire from there. Manual loop per prediction update.
- **Multi-template release matching.** Independent of predictors —
  capture release templates at e.g. 5 different sweep angles, fire
  on whichever matches first. Even before any predictor is fitted
  this gives the bot a wider effective firing window per sweep
  cycle, increasing throws-per-session and therefore data flow.
  Pairs with (a) above as the consumption path for a future
  predicted-angle output.
- **Variance-scaled adjustments** (port from hoops). When a darts
  predictor is fitted, use GP σ to decide how aggressive to be: high
  σ → either skip-the-shot-and-pick-a-different-angle or fall back
  to the captured default; low σ → trust the prediction. Less
  applicable in darts than hoops since darts lacks hoops' "hoop
  doesn't move on miss" constraint — players reset position on
  hits, so being picky about shots is less self-defeating here.
- **Click-timing audit on every minigame** — done for hoops and darts
  (2026-05-04 / 05). When adding a new minigame, audit the path from
  trigger detection to click landing for any disk writes or full-
  window grabs sneaking in. See CLAUDE.md "Click timing".
- **Tighten 9-dart streak detection to bullseyes only.** With the
  +1/+2/+3/+5 stripe scoring confirmed, a bullseye gives a +5 score
  increment while regular stripes give +1/+2/+3. Once score-OCR is
  wired (next item), the streak counter can require `score_increment
  == 5` instead of any hit. Eliminates the false-fire concern on
  STEAM_SCREENSHOT_ON_NINE_DART.
- **Lift run-time toggles into the launcher GUI.** Constants like
  `STEAM_SCREENSHOT_ON_NINE_DART`, `MONITOR_FLIGHT`, `MONITOR_MODE`
  are user-facing options today only via editing source. Once there
  are 2+ such per-minigame toggles, expose them as launcher
  checkboxes / settings (alongside or replacing the existing per-bot
  buttons in `ui/launcher.py`). Same idea would apply to hoops'
  `RESCUE_ENABLED`, `PREDICTOR_KIND`, etc.
