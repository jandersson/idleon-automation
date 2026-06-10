# TODO

Most open work is tracked as GitHub issues:
<https://github.com/jandersson/idleon-automation/issues>. Useful
labels:

- [`area:cleanup`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22area%3Acleanup%22) — refactors, dead-code removal, code follow-ups
- [`area:metrics`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22area%3Ametrics%22) — `shots.db` schema, OCR replacement
- [`minigame:hoops`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22minigame%3Ahoops%22) / [`minigame:mining`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22minigame%3Amining%22) — per-minigame work

`docs/cleanup_backlog.md` and `docs/metrics_backlog.md` keep the
historical record of shipped diagnostic columns and settled
experiments.

This file holds cross-cutting items, scheduled-agent IDs, and the
recent-changes snapshot that don't fit a single issue.

## Open

- ~~**Hoops scoring 10+**~~ — done 2026-06-10: drift measured from
  flight frames (~15-35 px/s horizontal sweep), `_should_refresh_hoop`
  re-detects the rim while the platform closes on the target and
  carries the offset to the fresh position; lost makes at 10+ (respawn
  signal is disabled there) recovered via the score-anchored retro-make
  (`made_source='score'`). Watch the 20+/30+ escalations for faster
  sweeps.
- **Poll-loop tick rate (darts + the hoops "tick mystery")** —
  profiled 2026-06-10 (`scripts/profile_poll_loop.py`): the darts loop
  sleeps only 20ms but delivered ~248ms median cadence because the
  tick is compute-bound — `find_game_over` 115ms (57% of the tick, run
  every poll), `find_release_pose` 72ms, mss grab 13ms, arm centroid
  2ms. "Increase polling" is therefore an optimization, not a setting:
  there is no idle time to reclaim. Progress:
  - ~~(1) normalize the dy signal to px/second FIRST~~ — done
    (64adbcd): vy px/s end-to-end, motion mask pinned to
    `REF_MOTION_DT_S` frame spacing, legacy rows convert via per-fire
    poll gaps.
  - ~~(2) cheapen `find_game_over`~~ — done: template caching +
    half-res coarse prefilter (gate 0.7 vs measured coarse floor
    0.48-0.54, true banner 1.000). 115ms → ~35ms; tick ~202ms →
    ~128ms measured, expected live cadence ~150ms.
  - ~~(3) cheapen `find_release_pose`~~ — done: `ScaleLockMatcher`
    (common/templates.py) polls only the locked scale ±1 step plus the
    junk-floor scale 0.6, full sweep every 20th poll. 78ms → 41ms;
    steady-state tick 88ms measured, expected live cadence ~110ms.
    Validated on 2029 recorded polls: 2000 bit-identical to the full
    sweep, worst conf delta -0.068, all disagreements junk-floor-side
    (scripts/validate_pose_scale_lock.py). NOTE: coarse-to-fine
    (half-res localization + full-res crops, the find_game_over trick)
    was tried first and REJECTED — the 9x40 release template is too
    small at half res (66/449 exact agreement, median conf -0.03,
    only 2.9x). matchTemplate cost here is ~11ms per scale regardless
    of template size (per-location overhead dominates), so fewer
    scales is the right lever, not smaller templates.
  - (4) only if game stutter is actually observed, a launcher "target
    tick ms" knob becomes meaningful as a CPU-budget control (cheaper
    ticks + a target cadence REDUCE load, not raise it).
  ~~Hoops' unexplained 200-1000ms tick~~ — explained and fixed
  2026-06-10 (`scripts/profile_hoops_loop.py`): same compute-bound
  template stack. Steady tick 179ms → 67ms via template caching,
  find_game_over every 5th tick (persistent terminal screen; no
  hoops game-over ground-truth frame exists on disk, so the darts
  coarse-gate trick can't be calibrated — the Nth-tick skip needs no
  calibration), and ScaleLockMatcher on find_platform (1199/1207
  bit-identical on the corpus). find_rim deliberately kept on the
  full sweep: clipped extreme-low-spawn rims match at scale 0.6 conf
  ~0.45 (the documented threshold case) and a lock would blind most
  polls to them. Side discovery, fixed same day: the pre-game prompt
  check had been running BETWEEN the fire decision and the click
  since 2026-05-09 (~104ms full-frame match) — see
  docs/hoops_findings.md fire-latency addendum.

- **Darts #40 — bust zones, new evidence (2026-06-10 23:30 session,
  not yet posted to the issue):** the yellow stripe became a bust zone
  late in the session and the bot logged the busts as generic misses.
  Throws 18/19 (apex_y 211/204, inc=0) flew arcs identical to the same
  session's yellow hits (throws 10/13: apex_y 207/212, +2 each); both
  busts hit the same zone and ended the session. Trigger does NOT look
  like live streak — streak was 0-1 at the busts (reset at throw 16)
  but score was 100+; yellow still paid +2 at streaks 10/13 earlier.
  So "high streak" may actually be "score threshold". Cheap mitigation
  without visual zone detection: a band the session has scored that
  suddenly returns inc=0 gets vetoed in `model_vy_band` for the rest
  of the session (one life instead of two). Proper fix: classify the
  stripes' bust state visually per throw. Also noted: the E[stripe]
  model currently trains busts as zero-score outcomes at yellow-band
  vy — directionally fine, but conflates board-miss with bust.
- **Chopping** — first instrumented session ran 2026-06-11 00:13
  (3 chops, 2 survived, ended ~4s in) and answered most of the open
  questions: leaf speed 257-386 px/s; red zones GROW per chop
  (r50→r53, r45→r48; green 120→114); the death was chop #3 firing
  with red 19px AHEAD of a rightward leaf (~50-75ms at that speed —
  click latency ate it) while chop #2's 8px of red BEHIND was safe.
  Fix shipped same night: directional time-to-red gate
  (`MIN_TIME_TO_RED_MS=150`, leaf vx in px/s from a 0.3s track;
  pixel margin kept as a direction-blind floor) + leaf_vx_px_s /
  red_ahead_px / time_to_red_ms logged per chop and poll. Needs a
  validation session. Still open: what ends a round naturally (timer
  vs chop count — every session so far ended by death or unknown),
  gold-zone value (gold appeared only in a transition layout so far),
  and whether 150ms is the right budget (tighten from outcome data).
  The leaf ACCELERATES over the round (game fact, 2026-06-11) — the
  vx-based gate self-adapts, but watch validation sessions for
  late-round fire starvation (rising speed + shrinking green leaves
  fewer gate-open windows); the polls' leaf_vx_px_s column records
  the speed ramp for calibrating the budget per speed tier.
  Community-mechanics research + per-attempt strategy (front-load
  chops, always take yellow, never click red) in
  `docs/chopping_notes.md` — headline: one red click ends the round,
  yellow pays +2 AND slows the leaf, chops shift the zone layout,
  and the inter-chop cooldown (~210ms) is the binding constraint on
  score, not aim.
- **Darts** — release-pose template matching works, score-region diff
  per throw works. Multi-template per spawn-height (the dominant
  accuracy variable) not yet built.
- **Predictor: Gaussian Process regression** — biggest jump available
  with the data we already have. Beyond just predicting platform_y,
  GP gives a *posterior variance* per query — the bot can say "high
  confidence, fire" vs "no nearby training data, skip this shot".
  That turns the predictor from a point estimator into a strategic
  signal. Reference implementation in own repo
  https://github.com/jandersson/gp_regression. Squared-exponential
  (RBF) kernel is the standard starting choice; lengthscale +
  variance hyperparameters fitted via marginal-likelihood
  maximisation. Add as `GpPredictor` in `common/predictor.py` +
  `PREDICTOR_KIND="gp"` switch. Likely needs sklearn or GPy; check
  pyproject.toml dependency policy first. New shot_log column for
  `predicted_uncertainty` so we can analyse the relationship between
  variance and actual hit/miss rate.

- **Predictor: LOESS / locally-weighted linear regression** — smaller
  upgrade on the current "KNN with inverse-distance" predictor,
  which is basically Nadaraya-Watson kernel regression with a hard
  K-nearest cutoff. Same weighted neighborhood, but instead of
  *averaging* the past makes' platform_y values, fit a small
  weighted lstsq through them so the predictor captures the local
  *slope* of the surface, not just the local height. Add as
  `LoessPredictor` in `common/predictor.py` + `PREDICTOR_KIND="loess"`
  switch in `minigames/hoops/main.py`. Uses the same `fetch_makes`
  data — no new schema. Cheaper to land than GP.

## Someday

- **Tighten `common.window.get_bounds`** so it doesn't match the first
  window containing the substring. Today it works because every
  minigame uses the unique "Legends Of Idleon" title, but if a future
  window title contains that string the matcher silently picks the
  wrong one. Could prefer exact match, or filter by process name.

## Scheduled agents

- **Weekly review** — `trig_01M2YTDKxnUyB9bYJpSZhiEc`. Fires every
  Saturday 10:00 Stockholm. Reads `shots_snapshot.json`, opens a PR
  with stats + suggestions if anything looks off. First run: 2026-05-09.
- **One-time deep review** — `trig_016jn83AvPZwZQJZjZpHHKgk`. Fires
  once on **2026-05-16 10:00 Stockholm**. Deeper analysis with
  predictor residuals.
- **Hoops #37 gate + #38 readiness check** — `trig_01ByQcCvVLxqGcpKM2vTjQqi`.
  Fires once on **2026-06-13 10:00 Stockholm**. Reads
  shots_snapshot.json; (1) computes the post-policy make rate (explore
  shots excluded) with a Wilson 95% CI vs the 25.5% baseline, posts
  the verdict to #37 and closes it if attempts >= 100; (2) checks the
  #38 data bar (>=50 dir=down shots in low+band regions with >=10
  makes, from the snapshot's vy_coverage) and posts status to #38.
  Note: part (2) was overtaken on 2026-06-10 — make_prob was promoted
  to default (ed4b841, 55/83 with CI clear of the policy stack), so
  the routine's #38 readiness post is informational only.

Manage at: https://claude.ai/code/routines

## Done (snapshot, May 2026)

The bulk of recent work happened in early May 2026. Major arcs:

- **Hoops detection rewrite** — rim-only template (was full-structure
  with stretchy pole), tighter score region, 20s exit timeout when no
  rim found.
- **Make detection** — replaced fragile `score_changed` diff with
  multi-pass Tesseract OCR + ball-trajectory cross-check. Anchored on
  a running `session_score` so per-shot OCR drops don't lose makes.
- **Predictor split** — `common/predictor.py` holds `KnnPredictor` and
  `BivariatePredictor` (selectable via `PREDICTOR_KIND` in
  `minigames/hoops/main.py`). Replaces the older bivariate-only model
  with hardcoded overrides.
- **Click-position experiments** — proved click position has no
  measurable effect on aim. Removed all related machinery.
- **Auto pipeline** — every session ends with a snapshot refresh +
  auto-commit + auto-push (within 09:00–22:00 Stockholm). Predictor
  retrains from the latest committed data on every launch.
- **Launcher** — Bots tab + Setup tab + Frames tab; tries-counter
  strip auto-reads from the local Idleon save (via plyvel-ci +
  vendored Haxe decoder).
