# Hoops findings — the June 2026 miss investigation (#37)

Settled conclusions from the 2026-06-09/10 investigation into why the
hoops bot's make rate was low (~25%) and inconsistent. Kept here so the
"why" survives outside the GitHub thread; the per-session evidence lives
in issue #37's comments. Follow-up model work: issue #38.

## The headline reframe: misses are not aim error

Shots whose ball arrives within ±120px of the hoop make at ~64%, and
made shots cluster within a 16px standard deviation. The bot does not
lose to near-misses — it loses to two categorically different failure
modes that both masqueraded as "way short" in the data:

- **Structure clanks.** The ball *reaches* the hoop (ball_peak_x within
  ~25px of hoop_x), clips the rim front / pole, and bounces backward —
  and `ball_x_at_rim_height` then measures the post-bounce descent,
  recording a 100–350px "short miss". Most of the historical way-short
  population was this. Any aim analysis must use the bounce-aware
  arrival (`_shot_arrival_x`: peak_x when backward drift exceeds
  `MAX_BACK_DRIFT_PX`, landing otherwise).
- **Genuine over/undershoots**, controlled by platform velocity at
  launch (below).

## Velocity controls the launch — through arc shape, not just range

The ball inherits the platform's vertical velocity at click time.
`platform_vy` (least-squares slope over the ~1.5s of platform samples
before the click; positive = downward) orders same-hoop outcomes
monotonically. Notes on interpretation:

- Near a bob turnaround the trailing window mixes the previous half-
  cycle, so window-vy is best read as a *phase proxy*: strongly positive
  on a dir=up fire means "just after the bottom turnaround".
- dir=up: slow launch (turnaround fire) → flat low arc (apex_y ~138)
  that reaches near hoops but clips the structure; fast launch
  (mid-swing) → high arc (apex ~81) with much more range.
- dir=down: slower descent at fire → longer flight (vy 28 → peak 887;
  vy 70–110 → peak 508–602). Monotonic and, unlike up, without a cliff —
  learnable structure.

The main-loop sample cadence is ~200–1000ms per tick (not the 15–30ms
the POLL_INTERVAL comment suggests) — anything derived from the sample
buffer must tolerate that.

## Direction is a per-hoop policy

`_required_direction_for(hoop_y, hoop_x)`; one predictor fitted per
direction (training rows are direction-filtered).

- **Very low hoops (y ≥ 530): dir=up is futile, dir=down works.** A
  56-shot exploration sweep at (691,550) covered the full bob range and
  every dir=up shot landed ≥68px past the hoop; the only reach-matching
  launch state is the bob-bottom flat arc, which clanks. The first
  dir=down shot at that hoop made (after 110+ dir=up misses).
- **The near "clank band" (x ≤ 640): bracketing confirmed.** dir=up
  arrives on target and clanks the rim front from every launch state
  (2/36 instrumented); dir=down undershoots by 50–110px with 2/12 makes
  while still cold-starting. The band boundary is genuinely 2-D —
  (659,350) clanks with up, (658,384) made with up — which is the kind
  of structure issue #38's model should eventually learn.
- Mid/far hoops (x > 640): dir=up makes at ~80% in the instrumented
  slice. Untouched.

## Make detection is multi-signal; OCR alone loses ~10% of makes

Post-shot score OCR drops out often enough that two safety nets exist:

- **Hoop-respawn signal** (`made_source='respawn'`): the hoop only
  teleports on a make, so a >30px hoop move after an unconfirmed shot
  retro-corrects the row. Guards: disabled at score ≥10 (drift moves the
  hoop by design), rim conf ≥0.9 required, clean_make re-classified from
  the shot's own trajectory.
- **Prompt-anchored score**: while the "Make a shot to start" prompt is
  visible the score is definitionally 0 — the session_score anchor
  resets and pre-shot OCR reads are ignored (a single misread "7" once
  disabled make detection for an entire game).
- **clean_make is never bypassed.** Prompt confirmation proves a shot
  *scored*, not that the aim was right; lucky pole-tip bounce-ins count
  on the scoreboard but stay out of predictor training (the old
  prompt exemption was a loophole, removed).

## Perturbation sweep: small, capped, hold-aware, cycling

- Capped at ±32px after σ scaling: every historical perturbed shot
  beyond ±32 landed >120px off with zero makes; the old ±240px swings
  were a miss spiral (perturbed shots made 14% vs 37% unperturbed).
- In-band misses (bounce-aware arrival within 60px) re-fire the same
  target — but at most twice in a row (`MAX_CONSECUTIVE_HOLDS`): at
  clank-prone hoops every arrival reads in-band and a hold loop once ran
  54 shots.
- On exhaustion the sweep cycles instead of pinning at the last entry
  (pinning fired 25 identical shots whose outcome was pure phase luck).
- Residual-proportional correction (aim off by the measured miss
  distance) was considered and **dropped**: landing does not respond
  coherently to offset in exactly the regions where correction would
  activate.

## Free-shot exploration

While the start prompt is up, misses cost no lives. The first shot at a
fresh hoop stays on predictor aim; after it misses, targets are sampled
uniformly across the observed bob range (`target_source='explore'`).
One 56-shot grind produced a complete dir=up landing map of a hoop —
the single most informative session of the investigation.

## Training-data hygiene

Trajectory predictors train on bounce-aware **arrival**, not raw
landing (`fetch_clean_trajectories`): floor-bounced shorts keep their
row with peak_x as the honest reach; hoop clanks are dropped as
reach-censored. The legacy launcher-distance filter both discarded the
short-mislaunch population (the velocity signal) and admitted bogus
post-bounce landings.

Schema columns added during the investigation: `platform_vy`,
`made_source`, `prompt_up`, `target_source`.

## Outcome

The measurement gate passed on 2026-06-10: 132 post-policy attempts
(exploration excluded), 46 makes — 34.8%, Wilson 95% CI [0.273, 0.433],
lower bound clear of the 25.5% baseline. #37 closed with that verdict.

## Open at time of writing

- ~~Measurement gate for closing #37~~ — passed (above).
- Make-probability model: issue #38.
- Loop tick rate (200–1000ms vs documented 15–30ms) unexplained; faster
  sampling would sharpen both vy and the crossing detector.
