# Fishing minigame — mechanics, bot plan, open questions

Research for the fishing bot (scaffolded 2026-06-17). Sourced from the
idleon.wiki *Fishing Minigame* page + the *Fishing* skill page's minigame
section; cross-checked against the existing bots' patterns. Pointer from
`minigames/fishing/main.py`.

## The mechanic

A **hold-to-cast distance game**, unlike the click-timing minigames:

- **Hold** the left mouse button (PC) to charge, then **release** to cast
  the lure. The hold duration sets how far the lure flies.
- Land the lure on a fish to score. Fish are distinguished by colour:

  | Fish | Colour | Points |
  |---|---|---|
  | Green Fish | green | 1 |
  | Eel | yellow | 2 |
  | Squid | purple | 3 |
  | Whale | blue | 5 |
  | Megalodon | red | trophy only (behemoth) |

- **Mines** litter the water. Key rule (wiki, emphasised): *landing on a
  fish still counts even if a mine is directly under it* — you only fail by
  landing on a **mine with no fish**. So the bot should aim at fish and
  treat mines as "don't land on a mine-only spot," not "avoid the whole
  area."
- **Difficulty ramps**: more casts → fish start **moving** and more mines
  appear. (So detection/aim must handle moving targets late-game.)

## Streak (unlocks higher-value fish)

Consecutive fish landings escalate which fish appear:

- 3 consecutive Green Fish → **Eels** start appearing
- +3 more consecutive landings → **Squid** appears
- +7 more consecutive landings → **Whale** appears
- **Catching the Whale resets the streak to 1.**

Megalodon (the trophy): catch two Whales in a row, then **miss** the third
cast → the "behemoth" appears; catching it gives the Megalodon trophy.
(Wiki-confirmed 2026-06-17; an earlier note here said "catch a different
fish" — it's a *miss*, not a catch.)

## Control model — hold_ms ↔ cast distance

The bot's one control knob is the hold duration. `cast_model.py` fits a
linear `landed_dist_px = slope·hold_ms + intercept` over logged casts and
inverts it (`hold_for_distance`) to aim. Linear is the starting form; if
observe data shows a charge **cap** or easing, swap in a monotone curve
there (callers use the inverse only).

- Fitted once at startup from `fishing.db` (the darts `stripe_model`
  pattern). `MIN_SAMPLES` clean casts gate the fit; below it the bot
  **explores** — random holds in `[EXPLORE_HOLD_MIN_MS, …MAX_MS]` — and
  logs `(hold_ms, landed_dist_px)` to build the curve. `EXPLORE_EVERY_N`
  keeps sampling after the fit.
- Target selection (`choose_target`): highest point value among fish the
  model can **reach**, nearest as the tie-break.

## Detection (HSV, like chopping)

Targets are colour-coded, so `detector.py` uses HSV masks + blob centroids
(the chopping gold>green>red style), one mask per fish colour + mine +
megalodon. **The HSV ranges are first-guess placeholders** — they must be
tuned against real frames (`fishing-calibrate` dumps per-colour overlays).

The **lure** (landing position), **play prompt**, and **game-over** screen
need sprite templates (`assets/lure.png` / `play_button.png` /
`game_over.png`), captured via `fishing-capture`. Until `lure.png` exists,
landings can't be measured, so the cast model can't train.

## Open questions (resolve from observe runs before trusting the bot)

1. **Cast geometry**: is distance purely horizontal (1-D), or a 2-D arc?
   Where is the cast origin (rod tip)? `_cast_origin` is a placeholder
   (play-region centre, lower third) and must be calibrated — target
   distance is measured from it.
2. **hold→distance shape**: linear, or capped/eased? Drives whether the
   linear `cast_model` survives or needs a curve.
3. **Does a mine-landing end the game, or just miss?** The wiki says
   "you're allowed to miss as much as you want" yet "you won't lose if you
   land on a fish" — ambiguous about a pure mine hit. Confirm; it decides
   how hard to weight mine-avoidance vs. the no-fish-timeout bail.
4. **Hold position independence**: assumed (per the CLAUDE.md
   "Idleon clicks are buttons" finding) — the bot holds at play-centre.
   Verify a hold elsewhere casts the same distance.
5. **Streak strategy**: a Whale is 5 pts but resets the streak. Is
   value-max optimal, or should the bot skip Whales to keep climbing /
   bank Eel/Squid? Decide once outcomes are logged (the #23/#38
   discipline). `choose_target` is value-max for now.

## Bot setup order (once the game is in front of it)

1. `fishing-pick-play-region` — bound the water.
2. `fishing-observe` — verify the detector sees fish/mines on real frames.
3. `fishing-calibrate` — tune `FISH_HSV` / `MINE_HSV` in `detector.py`.
4. `fishing-capture` — grab frames; crop `assets/lure.png` (+ play_button,
   game_over) so landings/game-over can be detected.
5. Run `fishing` — it explores holds, logs `(hold_ms, landed_dist_px)`;
   after `MIN_SAMPLES` the cast model fits and aiming switches to `model`.

## Observed structure (first observe session, 2026-06-17)

27 frames (960x572). The minigame is a **horizontal cast bar drawn over the
world**, player-anchored (like catching) — NOT a separate screen:

- A solid **blue bar** is the cast track; the lure is cast **rightward** along
  it. `N PTS` (current) sits at the left end, `N BEST` at the right.
- **Fish and mines are positioned along the bar.** Green fish is a green blob;
  mines are grey spiky balls with a red core.
- So the geometry is **1-D horizontal**: distance = target x − track start, and
  the **cast origin is the bar's left edge** (resolves open question 1).
  `find_cast_bar` detects the bar; `_cast_origin` now anchors there.

### Detection is confined to the bar (and still needs work — #63)

The bar overlays the world, so colour masks over the raw frame are flooded by
scenery: the **tan dock reads as `eel`** (H~13-30) and **shore plants as
`green`**. `find_fish`/`find_mines` now take the `find_cast_bar` bbox and mask
outside it. That fixes green (verified in-game H77-82) and squid, but inside
the bar the **score text** (`N PTS`/`N BEST`) still fires `eel` and the **mine
red-cores** fire `megalodon`, and blobs merge — the warm-hued fish need a
text-zone exclusion or template/shape match (tracked in **#63**).

### HSV (calibrated from the wiki sprites; green verified live)

| fish | pts | hue | note |
|---|---|---|---|
| green | 1 | H72-90 | verified in-game (H77-82) |
| eel | 2 | H13-30 | yellow; collides with score text |
| squid | 3 | H132-150, dark V | purple |
| whale | 5 | H100-120, **low S** | blue; S-capped to split from the blue bar |
| megalodon | — | H3-12 / 172-179 | red; collides with mine cores |
| mine | fail | low-sat grey body + red core | spiky |

The lure shares the warm-red hues, so it's matched by **template**
(`find_lure`), not HSV — `assets/lure.png` must be captured live via
`fishing-capture` (a wiki sprite won't match the live pipeline — the catching
play-button lesson).

## Aim off the charge bar — implemented closed-loop (2026-06-17)

The cast now runs **closed-loop on the charge bar** (`common.input.
charge_and_release`): hold LMB while polling `find_charge_level`, release at a
target fill. The model is `charge_level → landed_dist` (`cast_model`), aiming is
in charge-space (`charge_for_distance`), and the rod-not-ready state is detected
and skipped. This replaced the open-loop `hold_ms` cast that undershot near the
saturation cap and fought off-crop bobber noise. See the decision record below.

### Offline validation (run-7 saved frames, reconstructed)

The 7 live runs predate the `find_charge_level` commit, so every DB
`charge_level` is NULL — but `--save-frames` dumps from run 7
(`assets/captures/botrun_20260617_212435/`) let the charge be reconstructed and
paired with the logged `landed_x`. The relationship is clean and saturates:

| charge | landed_x | x / charge |
|---|---|---|
| 32 | 165 | 5.16 |
| 36 | 181 | 5.03 |
| 44 | 221 | 5.02 |
| 49 | 241 | 4.92 |
| 53 | 264 | 4.98 |
| 59 | 294 | 4.98 |
| 60 | 301 | 5.02 |

`landed_x ≈ 5.0·charge` (R²≈0.999); the per-cast reading is rock-steady (all
~14 landing-poll frames of a cast read the identical fill). Holds of 754/780/859
ms all plateau at charge ≈ 59–60 → `landed_x ≈ 301`, i.e. the **distance cap
(~267px from the bar-left origin) is the bar saturating at charge ≈ 60** — so
the cap is encoded by the model's charge support, not a separate term.

### The `charge_level=0` anomaly, diagnosed

Not random. Run 7's casts came in **pairs with identical `hold_ms`** (553,553 /
754,754 / …): the first charged and landed, the second read **charge=0 with no
bobber**. Cause — after a cast the lure is still out / reeling, so the next
`hold()` fired into a **rod-not-ready** state and built no charge, a wasted
cast; `choose_target` then re-picked the un-moved fish → same hold, hence the
pairs. The closed-loop cast fixes this directly: a not-ready attempt reads 0
through the grace window, aborts, and is retried — not burned as a logged cast.

### Left to validate live (the one thing offline can't cover)

The hold-while-polling **timing** is the only unproven part. On a fresh
`uv run fishing`, confirm: casts actually fire (charge bar seen at the play
crop's left edge — if the region's left edge isn't on the bar, every cast reads
0 and aborts, as run 6's frames showed), `n_not_ready` is moderate (rod-recovery
polling, not a stuck loop), and makes/points/streak climb. Then refit
`charge_level → landed_dist` from the now-populated DB and check the slope
matches the ~5.0 above.

## Decision record: closed-loop vs open-loop charge (2026-06-17)

**Decision.** Cast **closed-loop** — poll the charge bar while holding and
release at a target fill (new `charge_and_release` primitive) — rather than
keep the open-loop `hold(hold_ms)` and merely retrain the model on charge.

**Why.**
- The charge fill → distance map is clean, in-crop, and reproducible (table
  above); charge is the right control/predictor regardless of mechanism.
- Closed-loop is **robust to hold→charge drift**: it releases on the observed
  fill, so it doesn't depend on a hold-duration calibration that can shift with
  frame rate / game state.
- It **inherently fixes the rod-not-ready waste** (the every-other-cast
  `charge_level=0`): a not-ready cast reads 0 and aborts cheaply; open-loop
  can't tell a dead cast from a live one and logs garbage.

**Alternative considered — open-loop, retrain on charge.** Keep `hold(hold_ms)`;
model `hold → charge → distance` and add an explicit rod-ready check before
firing. Implications had it been chosen:
- Smaller change, no timing-sensitive primitive holding the button while
  grabbing frames.
- Would rely on `hold → charge` staying as deterministic as run 7 suggested
  (500→32, 610→44, 754→59, then flat at the cap) — untested across frame-rate /
  late-game load, and a drift there silently biases every aim.
- The rod-ready check (e.g. "no bobber visible and bar at 0") is **less direct**
  than the closed-loop abort, which reads the actual charging state.
- The offline finding that `hold → charge` *is* roughly deterministic means
  this path is viable, not dead — it's the natural fallback.

**Backtrack conditions** (when to revisit open-loop): if live runs show the
closed-loop hold-while-polling is unstable — e.g. the per-poll `grab_region`
adds too much latency so release overshoots the target charge badly, the button
sticks, or the poll cadence can't keep up with the bar fill. In that case the
data to switch back already exists: `hold → charge` was near-linear up to the
cap (run 7), so an open-loop `hold_for_charge` calibration is a drop-in. The
charge→distance model is mechanism-independent and stays either way.

## Correction: the charge bar is a thermometer LEFT of the cast bar (2026-06-17, runs 8–10)

The first three closed-loop live runs (8, 9, 10) cast **0 times** — every poll
read charge 0 and aborted, yet the bot visibly max-charged and flung the lure
(the abort's `mouseUp` after the hold *is* a cast). Three things were wrong in
the original model above, corrected here:

- **The mechanic is release-TIMING, not hold-duration.** The charge bar
  **oscillates** while held — up to max, back down, back up (player-reported,
  confirmed by the fill ramp). Open-loop run 7 worked only because its holds
  (500–780 ms) all landed on the **first up-sweep** (monotonic 0→max, peaking
  ~780 ms); hold 859 → 294 < the 301 peak was the start of the down-sweep. So
  "hold→distance clean" was just the first up-sweep.
- **`find_charge_level` read the wrong thing.** It reads red rows at x<10 of the
  *play crop*. The real meter is a **vertical red thermometer ~48px LEFT of the
  cast bar** (full-window x386–412, y170–252 with the cast bar at (434,233)),
  filling bottom-up. The play crop starts at window x432, so the thermometer is
  **off its left edge** — x<10 never sees it live. The run-7 "charge→distance"
  signal was a *post-release* left-edge bar that the player's position happened
  to put in-crop; during the hold there's nothing at x<10. The clean
  `landed_x ≈ 5.0·charge` was real numbers but a post-release proxy for the same
  hold-driven distance, not a live signal.
- **The capture was never stale.** A hold's 16 full-window frames are all unique
  (frame-to-frame diffs 200k–2M); the bar simply wasn't in the captured crop.

Fix (`find_charge_fill`, commit reading the thermometer): grab the **full
window** each poll and read the thermometer's red-fill height anchored to the
detected cast bar (`CHARGE_BAR_DX/DY` offsets, tuned on run-10 frames →
empty=0, full=56, clean monotonic ramp). The closed loop releases at a target
fill on the up-sweep — the right control for a timing game. `find_charge_level`
is kept for the post-release read / back-compat. The fill→distance model still
needs fresh data: runs 8–10 logged no real casts, so the bot explores target
fills and logs `(charge_level=fill_at_release, landed_dist)` until it fits.

## Catches are detected by the fish DISAPPEARING (2026-06-17, run 14)

A caught fish is consumed and **vanishes** the instant the lure lands on it. The
original made-detection (`kind_at` at the landing, post-cast) therefore scored
every catch as a miss — there's no fish left to find. Run-14 cast 6 (saved
frames): the green fish sat at x109 in every frame until the lure landed at 107,
then disappeared (a clear catch), logged `made=0`. Across that run, exactly the
two casts where a fish vanished next to the lure were catches; the bot caught 2
but logged 1, which also starves the streak (no consecutive catches → eels/squid
never unlock).

`_classify_catch` now compares the fish near the landing in the **pre-arrival**
frame vs the **landed** frame: present-then-gone = caught (its kind); still
sitting there = the lure missed it (cast 3: fish at 203, lure overshot to 226).
`CATCH_RADIUS=15` px (observed catch offsets ~2–11px). Caveat: it's a
single-frame pre/post comparison, so a fish-detection flicker could mis-score;
the score-text OCR (`N PTS` on the bar) would be a stronger cross-check if this
proves noisy. The fill→distance model is unaffected (it trains on landed_dist,
not `made`) — this fixes points/streak reporting and unlocking higher fish.

## Open accuracy notes (runs 11–14)

- **Distance saturates ~fill 44** (dist ~210): beyond it the lure goes off the
  play crop and landings scatter (46→157, 50→156, 53→226). The reliable range is
  ~fill 15–44. Near/mid model casts land within ~2–10px; far casts overshoot.
- **The fish are stationary** (the detector reads a constant x across a cast's
  landing-poll frames) — no moving-target lead needed.
- **Cross-run offset**: run 13's casts landed ~25px shorter than run 12's for
  similar fills — likely a cast-origin/player-position effect; adds noise to the
  fill→distance fit. Worth pinning the origin more robustly if it persists.

## Sources

- IdleOn Wiki — Fishing Minigame: https://idleon.wiki/wiki/Fishing_Minigame
  (sprites: MGgreenfish/MGyellowfish/MGpurplefish/MGbluefish/MGredfish,
  MGfsh5 mine, MGlure)
- IdleOn Wiki — Fishing (skill), "Fishing Minigame" section: https://idleon.wiki/wiki/Fishing
