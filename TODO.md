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
  - (3) `find_release_pose` is now the dominant cost (~78ms; 7
    full-scale matchTemplates over the left 65%). Coarse-to-fine or a
    scale-cache would halve the tick again, but it's the
    accuracy-sacred path — separate change, validate against the
    recorded pose conf floor.
  - (4) only if game stutter is actually observed, a launcher "target
    tick ms" knob becomes meaningful as a CPU-budget control (cheaper
    ticks + a target cadence REDUCE load, not raise it).
  Hoops' unexplained 200-1000ms tick (docs/hoops_findings.md open
  item) is almost certainly the same per-tick template-match stack
  and the same fix order applies — faster sampling there sharpens vy
  and the crossing detector, the named miss mechanisms from #37.

- **Catching minigame** — scaffold only; detectors return None. Need
  fly + hoop-gap detectors before this runs.
- **Chopping** — last verified state was "button click is suspect".
  Hasn't been touched since the early-May Hoops focus push. Needs a
  fresh look + a careful single-attempt re-test.
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
