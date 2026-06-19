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
  landing-poll frames) — no moving-target lead needed. **SUPERSEDED (2026-06-18,
  see "Targets slide" below): the targets DO slide, starting ~cast 5; runs 11–14
  just ended before the slide set in.**
- **Cross-run offset**: run 13's casts landed ~25px shorter than run 12's for
  similar fills — likely a cast-origin/player-position effect; adds noise to the
  fill→distance fit. Worth pinning the origin more robustly if it persists.

## The "eels" were mine cores (#63, 2026-06-18)

Surveying 1278 saved frames: the eel HSV fired 83 times, **every one in a run
that never reached streak 3** — so no real eels were possible. It was catching
two things, neither an eel:

- **Dull warm scenery** — the tan dock, the red/white bobber, the score text —
  all at S~121. A real eel is saturated (S~197). **Raising the eel S floor
  60→150 cut 83→4.**
- **Mine cores.** Mines are spiky balls with **red spikes and a bright orange
  core**; the orange core (S>150) passed even the raised floor. The old
  `MINE_HSV` looked for a *grey* body and matched ~0/1278, so mines were
  invisible and their cores read as eels.

Fix: detect mines off their **RED spikes** (an orange mask floods the frame —
the bar overlays an orange sunset sky), with a compact size gate (~24–52px ball
vs ~17px fish). `find_fish` then drops an eel/megalodon blob that sits inside a
detected mine (the core), while **keeping** a green/squid/whale on a mine (those
are catchable). Net on the corpus: green 1166 kept, eel false-positives **0**,
mines detected in ~840 frames.

Consequence for the cast-15 case: the `eel@129` was a mine core, now excluded —
so that overlap no longer scores a phantom eel (it falls to a miss, since the
green that was actually caught was fully occluded by the mine and isn't
detectable from those frames). Mine *avoidance* (don't cast at a mine-only spot)
is now possible since mines are detected.

### The real eel is a curled fish — matched by template (run 18, cast 10)

A clean, un-occluded eel finally showed up: a **yellow CURLED fish**. It defeats
colour detection two ways — its hue/sat (H12-24, S~122) is the **tan dock's**
(S~121), so a relaxed colour gate floods 68% of no-eel frames; and the curl
gives **fill ~0.20**, far under the square-fish gate. So eel is dropped from
`FISH_HSV` and matched by its distinctive **curled shape** via template
(`find_eel`, `assets/eel.png`, cropped from cast 10) — like the lure. Template
match **1.00** on the eel (0.95 as it swims) vs **≤0.63** across 263 no-eel
frames; threshold 0.75. Corpus-wide it fires **only in the two streak-3 runs**
(13 + 14) and zero in every no-streak run. `choose_target` can now target eels
for their 2 points.

**Two more poses added (botrun_142848, 2026-06-18).** One template wasn't enough:
on a streak-4 session the single pose matched **0.745** at best and detected **0
of the session's eels** (the user saw eels go un-fired-at). `find_eel` now matches
ALL `assets/eel*.png` and keeps the best; `eel_2` (a tight spiral curl) and `eel_3`
(a compact vertical curl) were cropped from that session's near-misses. Corpus
result on those 309 frames: **0→4 casts detected** (cast05–08), the new poses
GENERALISE (cast07/08 weren't templates yet match `eel_3` at 0.94), and **zero
false positives** — every non-eel cast (mines included) stays **≤0.633**, far
below 0.75. Add more `eel_<tag>.png` crops if other curls under-match (#63).

### First live squid + eel cluster (cast09, botrun_120520, 2026-06-18)

A streak finally surfaced a **squid** (purple) sitting beside an **eel** (yellow
curled) and a **mine** — squid partly overlapping the eel. Findings:

- **Squid HSV is validated.** `FISH_HSV["squid"]` (H132–150, dark V) — until now
  sprite-derived and unverified — detected the squid as `'squid'` in **5/6**
  pre-lure frames (x≈247), no false positives. The provisional range holds on a
  real squid.
- **Eel under-matches under occlusion.** With the squid overlapping it, the eel
  template scores **0.633** — just under the 0.75 threshold — so `find_eel`
  returns None even though the eel is clearly there. Part one-pose, part
  occlusion. (Eel *catches* are still recorded by the score +2 delta regardless;
  the miss only costs eel *targeting* this frame.)
- **The squid wasn't caught though reachable-now.** It sat at x≈247; the model
  reach is 75–283px now, but at this run's startup the reliable edge was ~210px,
  so the squid was likely just beyond reach at fire time — `choose_target`
  correctly took the reachable green (x70) instead. As the cast model's far
  support firms up, reachable squids (3 pts) should start getting targeted; worth
  watching that high-value fish near the reach edge actually get chosen.

## Background-invariant fish detection — masked-ZNCC sprites (#75, 2026-06-18)

Absolute-HSV color gates break per biome because the minigame is an overlay over
a **live, varying world**: the tan dock read as `eel` (#63), the turquoise beach
water flooded `green` and left the bot stuck on the no-fish path (#72). The fish
SPRITES are identical pixels regardless of biome — only the world behind them
changes — so detection should match the **sprite**, not its colour-vs-background.

`detector.find_fish_sprites` does per-kind, multi-pose **masked-ZNCC** matching
over the cast-bar band: a masked `TM_CCORR_NORMED` sweep localizes candidate peaks
(iterative max + neighbourhood suppression → multiple fish), each confirmed by
`masked_match_confidence` (ZNCC over the sprite's masked pixels) at
`SPRITE_ZNCC_THRESHOLD=0.55`. The mask drops the world behind the overlay, so the
same fish scores ~0.9 over any biome. Templates: `assets/fish_<kind>_<n>.png` +
`_mask.png`, bootstrapped from the existing corpus (green×2, squid×2 poses).

Validated offline (the cross-biome feasibility that justified the rebuild):
- masked ZNCC at the true fish location — **same biome 0.86–0.92, turquoise beach
  0.90, no-fish background 0.27–0.32** (clean ~0.9-vs-0.3 separation across
  totally different biomes; the JPG-shifted screenshot still matched at 0.88).
- Integrated `find_fish` over the corpus: **99% of HSV greens + 100% of HSV
  squids recovered, +8 fish HSV had MISSED found, 0 true false positives.** A flat
  green rectangle (which HSV calls a fish) is correctly rejected (not the sprite).

Merged **additively** into `find_fish`: a fish counts if HSV OR the sprite fires.
The dedup is **kind-aware** — a sprite det suppresses only a SAME-KIND HSV
duplicate, never a different kind. This is load-bearing: a proximity-only dedup
dropped a converged eel(2)/whale(5) sitting within `SPRITE_DEDUP_PX=12` of a
green/squid sprite, erasing the high-value fish from `choose_target` in exactly
the converged-cluster regime where it matters (caught by a multi-agent review).

Scope + open items:
- **Whale** has no sprite (never captured, #69) → stays HSV-only; provisional.
- **Eel** is now a masked-ZNCC sprite too (#77, done): its colour == the warm dock,
  so HSV can't mask it, but a **GrabCut body mask** (corners=bg, centre=fg) segments
  the curl, and masked ZNCC matches its pixels. This fixed the occlusion under-match
  (the squid-overlapped eel: 0.63/undetected unmasked → ~0.97 masked) and recovered
  52 eels the old path missed. `find_eel` is now a thin wrapper over
  `find_fish_sprites`. The partial `eel.png` ("half an eel") was the bad data behind
  the old 3-pose struggle and is dropped (`fish_eel_1/2` are full eels). Cross-kind:
  the green template structurally matches eels at ~0.82 (shared fish silhouette
  under ZNCC), but the highest-confidence-per-location dedup keeps the eel (~0.98) —
  244/244 eel frames labelled correctly. Eel masks come from GrabCut (not HSV like
  green/squid) precisely because the eel's colour == its background.
- Threshold 0.55 sits ~0.04 below the validated true-fish floor; safe given the
  ~0.20 gap to background and the now kind-aware dedup (a spurious green/squid
  sprite can no longer delete an eel/whale), but tunable if a new biome shows a
  phantom green/squid.
- **HSV is now RETIRED for green/squid** (done): `find_fish` skips any kind in
  `SPRITE_KINDS`, so green/squid/eel are **sprite-only** — fully biome-invariant
  (no absolute-hue path left for them). `whale` is the only kind still on HSV (no
  sprite captured, #69) — the last biome-dependent fish detection. Cost: on the
  single-biome corpus, sprite-only keeps **99.1% of green / 98.7% of squid** vs the
  prior additive (sprite|HSV) — the ~1% HSV-only margin, traded for biome-invariance.
  The green H88 cap (#72) is now historical (HSV green no longer runs).

## PTS score delta is the ground-truth catch signal (#58/#63, 2026-06-18)

The bobber-disappearance / eel-absence heuristics miss catches they can't see:
far casts that never settle a measurable bobber, and the curled eel (one template
pose, prone to under-match). The game's own **"N PTS" counter** doesn't lie — read
it just before a cast and just after the landing, and the **delta is the catch**:
+1 green, +2 eel, +3 squid, +5 whale, 0 a miss. `score_before`/`score_after` are
read from the FULL window (the counter renders below-left of the cast bar, left of
the play crop), anchored to the detected cast bar (`score.py` `SCORE_DX/DY`).
`_resolve_catch` trusts the two reads **asymmetrically** (so it only ever *adds*
correct catches, never fabricates a catch OR a streak-resetting miss):

- **delta > 0** (a known fish value) → authoritative catch: the score rose, so
  something was caught — the rescue for far-cast / eel catches with no bobber.
- **delta == 0** (a clean miss) → the counter can lag the catch by an animation
  frame, so a zero read right after the landing may be *stale*. It does **not**
  veto a catch the bobber/eel heuristic positively confirmed (that would
  fabricate a miss and reset the streak — the dangerous failure, flagged by the
  pre-commit review). The zero is recorded as a miss only when the heuristic
  didn't confirm a catch (where `_measure_landing` waited the full settle, so the
  read is reliable).
- a **None** read (uncaptured digit / unreadable) or an ambiguous delta → the
  heuristic verdict stands.

Logged per cast: `score_before`, `score_after`, `detect_source`
(`score`/`landing`/`eel_absence`).

### Binarization: white-fill, not Otsu (the dock bridges the digits)

The shared reader (`common/score_digits.py`) defaults to Otsu-minority binarize,
which is clean on water/sky but **bridges** on the busy tan dock — the dark plank
grooves binarize alongside the dark glyph outlines and connect the digits into the
scenery and the "PTS" label (components span the full crop height; the leading
digit can't be read). The score digits are a bright near-white **fill**, so keying
on that — `binarize_white_fill`, V≥190 & S≤110 — rejects the dock (saturated), the
planks/outlines (dark), and the coloured bar, leaving clean glyphs. The reader is
parameterised (`make_pts_reader(dir, binarize=…)`); catching keeps the Otsu
default unchanged. Validated on the botrun_225352 frames: **163/176 read "0 PTS"
→ 0, zero misreads** (13 had no cast bar → `None` → fallback); "26 BEST" → 26; the
P/T/B/E/S/T label fragments all score ≤0.35 vs the 0.6 match floor.

Two **noise blobs** are cropped out, one per axis. Left of the digit: the red
charge thermometer's bright highlight at `bar_x-38..-28` (present only mid-charge;
the digit's left edge is at `bar_x-21`) — `SCORE_DX0 = -26` starts the crop past it
with a ~5px margin. Below the digit: intermittent bright scenery (foam/splash) at
the crop's bottom-left forms a digit-sized component that becomes a spurious
leading digit (live botrun_120520: it broke ~1/3 of reads) — `SCORE_DY1 = 23`
crops to the digit band (the digits sit in the top, ~rows 2–10) so the bottom
noise is filtered as too short. With both, the leftmost component is the lone
leading digit. (Both were invisible in the bootstrap charging frames, which is why
the first cut used the full region.)

### Digit templates: bootstrap + capture flow

The reader matches each white-fill glyph against `assets/digit_templates/<d>.png`,
captured **through the same white-fill pipeline** the live reader uses (the
catching lesson — a template from another pipeline mismatches). 0/2/6 were seeded
from botrun_225352 ("0 PTS" / "26 BEST"); 1/3/4/5/8/9 were added from the
botrun_120520 live run (score climbed 0→16); 7 from botrun_123328 ("7 PTS"). The
**set is now complete (0–9)** — both live runs (46 crops) read monotonic with no
None and no misreads. The capture flow: `uv run fishing --save-frames` saves
each cast's before/after PTS crop to `assets/captures/botrun_*/score_*.png`, then
`fishing-capture-digits` dumps the isolated glyphs + a contact sheet to label and
copy into `digit_templates/`.

An **incomplete template set misreads, it doesn't just blank**: with only 0/2/6
present, live 3/8/9 all false-matched "6" (the closest survivor above the 0.6
floor) — verified on botrun_120520, where the old reader read 6 for true 3/8/9 and
the full set reads each at 1.0. That's lossy but still SAFE: the collapsed reads
produce delta==0 or unmapped deltas, which the asymmetric `_resolve_catch` turns
into missed/None outcomes, never a fabricated catch. Still, capture the full set
(grab 7) so the score signal is trustworthy on every frame, not just those whose
digits are all captured.

### Background-invariant reader — masked-ZNCC glyphs (#82, 2026-06-19)

The white-fill binarizer above is **position-invariant but not background-
invariant**, and that broke when the player moved areas. The board POSITION is
already invariant — `find_cast_bar` locates the bar anywhere (verified on a beach
frame where the bar sits at x=60 vs the calibrated x=434) and the score crop
anchors to it — so the gap was purely the READER. After an area move the digits
render over **pale-desaturated turquoise water** (S~80–100, bright); that passes
the white-fill gate (V≥190 & S≤110) and **floods** the binary, drowning the
digits → `read_score` returned None. Coverage was bimodal by area: 100% in the
calibration spot, 0–12% across ~25 sessions after the move (`bar_screen_x/y`
logging, 1927a4e, pinned it to area, not biome). **No fixed threshold fixes it**:
`S≤60` reads the beach but regresses 36 dock reads (dock digit *edges* are S
60–110, overlapping the pale background), and dilating to recover thickness
merges digits. The binarize-then-match architecture with an absolute threshold
was the wall.

The fix mirrors the move that made fish detection biome-invariant (#75): match
each digit by its **glyph pattern via masked ZNCC**, dropping binarization. Per
digit (`common/score_digits.py`, `read_pts_zncc`):

- **Template = the grayscale glyph** (bright fill ~255 + dark outline/interior
  ~16–26); **mask = the ink** (the bright fill dilated 1px to grab the outline).
  The decisive observation: inside the glyph footprint every pixel is *intrinsic*
  to the rendered glyph — on the beach the "0" hole/outline read 4–26 (dark), the
  fill 255 (bright), and only the bbox **corners** show background. So a mask
  hugging the ink is fully background-invariant; the world only shows outside it.
- **Localize + match together** by sliding each template over the crop
  (`common.templates.masked_zncc_map` — the vectorized exact-ZNCC map, three
  `TM_CCORR` correlations, verified == `masked_match_confidence` to ~1e-4). Keep
  every peak ≥ `GLYPH_ZNCC_THRESHOLD=0.7`, dedup overlapping peaks across digits
  by x (highest ZNCC wins), assemble the leading left-to-right run, and **break
  at the first large x-gap** so a stray match in the "PTS"/"BEST" label can't
  extend the number. No foreground segmentation — the background-dependent part.
- **Threshold 0.7 sits in a wide gap**: true digits peak at ZNCC ~0.97+ (p1 over
  882 labelled instances = 0.971) while the label letters and the space-before-
  PTS score ≤0.62. The thin **"1"** is the only false-positive risk (it matches
  the "P" stem / a gap at z~0.56–0.62); the high threshold rejects those — at
  z<0.65 a spurious trailing "1" produced "151" for true "15".

Validated (`scripts/validate_fishing_glyphs2.py`, 1530 dock crops + the failing
beach frame): the beach reads **0**; vs the white-fill reader **0 value→different
regressions, 661 same, 841 recoveries**, all spot-checked correct (incl. 2-digit
"19"/"13" and a busy noisy background). The one value→None (`c31_after`) **corrects
a wrong old read** (old "1" on a frame whose score was 16 — impossible). Old-vs-old
comparison alone is insufficient (hallucinations hide in the recovery bucket where
old=None), so correctness was also checked by **per-session monotonicity** (a
session's score never decreases): **0 non-monotonic reads at 0.7**.

Templates are grayscale `assets/digit_glyphs/<d>.png` + `<d>_mask.png` (median
over many corpus instances), built by **`fishing-build-glyphs`** (bootstraps
labels from the white-fill reader on the dock corpus, where it works). The
white-fill path (`binarize_white_fill`, `digit_templates/`) is retained for
reference and still drives catching (Otsu default); `fishing-capture-digits` is
its capture tool. Fixing the reader also **restores squid/whale detection in
non-calibration areas** — they're attributed via the score delta, so a blanked
score lost them (#82).

Open refinement: digit matching is **scale-1.0 only** (all data is 960×572); a
window resize would shrink the glyphs and break the read. The cast-bar height is
a ready scale reference for a multi-scale digit sweep — filed as a follow-up.

### Open / to validate live

- **Multi-digit PTS layout is left-aligned** — CONFIRMED on botrun_120520
  ("10 PTS"…"16 PTS"): the leading digit's left edge stays fixed at `bar_x-21` and
  the number grows rightward (PTS shifts right), so `DX0=-26`/`DX1=45` capture
  2-digit scores cleanly. (A rare per-frame anchor jitter can shift the crop ~2px
  and clip a read → a negative/None delta → safe fallback; seen once in 62 crops.)
- **Score-update lag.** The after-read happens once `_measure_landing` returns —
  on a measured catch that can be only ~0.2–0.4s after the landing (early exit on
  the bobber reeling in), within the window where the counter may not have ticked
  yet. `_resolve_catch` handles this by not letting a `delta==0` veto a confirmed
  catch (above); the residual assumption is that on a far cast (no bobber, full
  ~1.6s wait) the score has settled, so its `delta==0` reliably records a miss.
  Confirm from live runs that measured `delta>0` catches read on time (the rescue
  path) and that far-cast zeros aren't lagging.
- The eel template is still one pose — the score delta now catches an eel (+2)
  regardless of the template matching, which is the point.

## Player position — anchor the crop to the bar, not a fixed rectangle (#58, 2026-06-18)

A session scored 0 after the player stood **slightly differently**. Cause: the bot
ran `find_cast_bar` on a FIXED `regions.json` play crop (a window-fraction
rectangle). The minigame UI is player-anchored, so a small move shifted the cast
bar until its **left edge ran off the crop's left edge** (the calibrated crop had
only ~2px of margin: bar left at full-x 434, crop left at 432). `find_cast_bar`
then returned a CLIPPED bar (width 278 vs the true 304, wrong left edge) — and the
charge thermometer (`bar_x-48`), score region (`bar_x-26`), and cast origin all
anchor to that edge, so they read garbage. Charge fired at a constant ~19
regardless of aim; score read nothing; every cast missed.

Fix: detect the bar on the **full window** (its true position, never clipped),
then derive a **dynamic play crop anchored to the bar** (`_play_crop_from_bar`,
clamped to the window) — `frame` is a slice of the full window, `bar` its position
within the crop, and `cast_bar_full` reconstructs to the true bar exactly. Now the
crop follows the player. Verified offline on the calibrated frame AND a simulated
−40px player shift (both: correct anchor, score, charge, fish); a multi-agent
adversarial review found no correctness defects. The `regions.json` "play" crop is
no longer read by the bot (it self-anchors) — `fishing-pick-play-region` is only
needed by the observe/calibrate tools now.

## Converged multi-catch — a delta is a SUM, not one fish (#58, 2026-06-18)

The sliding mechanic has a second consequence: fish **slide together and
converge**, so one cast can catch **several at once**. A run logged a "whale"
(+5) that was actually an **eel (2) + squid (3)** caught together as they
converged (user-confirmed from the scene: a green, an eel, a squid, two mines —
no whale). So the score DELTA is the *total points*, which can be a SUM.

Consequences + fix:
- `kind_from_delta` no longer maps +5→whale. It names a **lone** common fish only
  for green/eel/squid (delta 1/2/3); any other positive delta (4, 5, 6, …) is a
  **converged multi-catch** labelled `'multi'` worth its full delta. (Whales are
  rare and undetected, and the eel+squid sum is the common cause of a +5 — so a
  real whale is also logged `multi` until whale *detection* can confirm one.)
- **Multi-catch deltas were being DROPPED**: +4/+6/+7 returned None (no single
  fish) → fell to the heuristic → often unrecorded. Now they're real catches
  (made=1, points=delta). The points always come from the delta, so `'multi'`
  scores correctly.
- The streak no longer resets on a delta `'whale'` (that was a converged
  multi-catch mislabel). It also explains the **streak-counter undercount**: the
  game advances the streak *per fish*, the bot *per cast*, so a converged catch
  is +2 in-game but +1 to the bot's (cosmetic) counter — which is why squid/whale
  appeared at low logged streaks.
- **Caveat on old data**: pre-fix `landed_kind` counts (the "1 whale", some
  "squid") are unreliable — some were converged multi-catches. Points/`made` were
  always right (the delta); only the kind labels were.
### Vanished-fish kind attribution + per-fish streak (#66/#70, 2026-06-18)

The delta alone can't name a multi-catch or tell a lone eel from two greens
(both +2). On a **measured** catch the detector already knows which fish were
near the landing before vs after, so the fish that **vanished** are the actual
catch — `main.vanished_fish` returns that set. Matching is by kind **and
position**: each near-landing pre-fish is paired to the nearest still-unclaimed
post sighting of the same kind within `SLIDE_BUDGET_PX` (it slid, wasn't caught);
a pre-fish with no such match vanished → caught. The position pairing (not
kind-alone) is what a pre-commit review flagged as load-bearing — fish routinely
slide >`CATCH_RADIUS` between frames (see "Targets slide"), so a kind-only test
miscounts a slid-away fish as caught. The budget is set to the documented max
per-cast slide because over-pairing only costs a safe fallback while under-pairing
corrupts the data. `main.attribute_catch` then **cross-checks** the vanished
set's `FISH_VALUE` sum against the score delta and, on a match, names the kind(s)
(`'eel+squid'` instead of the opaque `'multi'`, sorted/joined) and returns the
**fish count**. Only scoring fish (value > 0) count, so a non-scoring kind can't
inflate the count or match coincidentally.

The residual the cross-check can't catch (review finding, accepted): an
undetected-but-caught fish (e.g. an under-matched eel) whose missing value is
exactly offset by an equal-value fish that flickered out of detection can pass
`sum == delta` on a wrong set. It needs a conjunction of detection failures, only
mis-labels data, and never touches `made`/`points`/streak-gating — so it's logged
as a known limitation, not guarded further.

Two consequences, both data-quality (kind still drives no decision):
- **landed_kind** is refined to the named set on delta-confirmed measured catches.
- **The streak now advances by the fish COUNT** (`n_caught`), not +1 per cast —
  fixing the #70 undercount where the game advances per fish but the bot counted
  per cast. Far casts / eel-absence / heuristic catches (no measured fish frames)
  keep the +1 estimate, and the value-sum cross-check falls back to the delta's
  kind guess + `n_caught=1` whenever it can't be trusted (flicker, a fish slid out
  of radius), so it only ever *improves* the label/count, never fabricates one.

Safety: the attribution touches **only** the cosmetic `landed_kind` and the
streak count — `made` and `points` still come solely from the authoritative
delta (`_resolve_catch`), so it can't fabricate a catch or a streak-resetting
miss. Pure helpers unit-tested in `tests/test_fishing_landing.py`; the live
pre/post fish sampling stays manual. (A real lone whale, if ever detected, still
attributes as `'whale'` and resets the streak; a converged catch *containing* a
whale is not special-cased — whales are undetected, so this is moot today.)

- **Caveat on old data stands**: pre-fix `landed_kind` counts remain unreliable;
  no backfill (the attribution runs live, not retroactively).

## Close-to-the-dock fish — reach floor, not detection (#58, 2026-06-18)

A fish spawning close to the dock went uncaught and the bot **flung the lure far
into a mine** instead. It's a REACH bug, not detection: the fish IS detected (e.g.
a green at x≈48–50 from the origin), but the cast model's reach floor is ~69px
(charge_min≈11), so `choose_target` deems it unreachable → returns None → the old
fallback fired a *random far* explore (charge 39 → ~207px, toward the mines).
Confirmed on botrun_131146 cast14 (green@50, mines@139/214, lure→207).

Fix: when `choose_target` finds nothing in reach but fish ARE present, aim SHORT
at the nearest one (`nearest_fish_target`) by extrapolating the charge down to
`EXPLORE_CHARGE_MIN` (`charge_for_distance(..., min_charge=)`), tagged
`aim_mode='nearest'` — the lure stays near the dock (safe even if the linear fit
drifts at low charge; the score delta still records a catch if it lands), never
flung far. And `EXPLORE_CHARGE_MIN` 10→5 so exploration samples short casts and
the reach floor drops over runs, turning these into ordinary in-reach model casts.

## Targets slide — the moving-target lead (the streak stall, #58, 2026-06-18)

The streak stalls at ~3 (so eels/squid/whale rarely unlock, and the run is all
greens). Cause, measured from the saved botrun frames + `fishing.db`: **the
targets SLIDE back and forth along the bar**, starting ~cast 5 and growing in
speed/amplitude through a game. Within one cast the motion is monotonic/linear
at **~±15–22 px/s** late-game, then it settles at a turn point.

**Mechanic confirmed (user, 2026-06-18):** each target oscillates within a
**FIXED RANGE and REFLECTS at both end-limits** — a constant-speed 1-D billiard
bounce (triangle wave), not a decaying drift or a sinusoid.

**Measured from 409 tracked casts + the lead telemetry (#74):**
- **Onset + growth:** targets are STATIONARY for casts 1–6 (median window-span
  0px), motion onsets ~cast 7, amplitude ramps and plateaus ~cast 14 at a
  median span ~10–13px (max 23). The range grows through the game, then plateaus.
- **In-flight reversals are RARE: 1/409** — the bounce period is long vs the
  ~1.3s flight, so within a single cast the fish moves ~monotonically ~10–20px.
- **The `MAX_LEAD_PX` cap was the binding problem, not the missing reflection.**
  Applied-lead velocities ~8–15 px/s give an intended lead `v*1.3s` ~16–20px, so
  the old 12px cap was BINDING on **88%** of leads — under-leading by ~4–8px at
  the ~10px catch-radius scale (a likely reason the lead was ~neutral in #67).
  Raised **12 → 18** to match the measured displacement; a reversal inside one
  flight is rare enough that the cap can safely track the real slide. Re-validated
  via the `model_lead` A/B.

A full **bounce-aware** lead — predict the folded triangle-wave position at
landing from (x, v, direction, the two limits, flight time) — remains the
principled fix for the rare in-flight reversal (#74), but is lower-value than the
cap raise: reversals are ~0.25% of moving casts, and the range grows through the
game so the limits must be estimated LIVE (the ~1.4s landing-poll window captures
a full turn-to-turn swing only 1/409, so even measuring the range needs longer
continuous observation than is logged today).

The bot aimed at
the fish's DECISION-time x with zero lead, but ~1.3s elapses from decision to
landing (closed-loop charge hold ~0.6s + lure flight ~0.7s), so the fish slides
**14–32px** off the lure by the time it lands — past the ~10px effective catch
radius. (Cast-distance error alone is only ~0–10px; the slide is the dominant,
fixable miss.) Per-cast trajectories: c18 fish 194→182 vs lure 203; c22 143→131
vs 163; c27 88→74 vs 94 (cast accurate to the *stale* target, fish gone).

### The lead — shipped logging-first as a parity A/B

`cast_model.lead_fish_dist` aims at the fish's PREDICTED landing position; for a
single fish the bot samples its slide velocity over a few quick polls before
firing (`_sample_fish_velocity` → `theil_sen_velocity`). Design vetted against
two failure modes a naïve lead would hit (a multi-agent design+adversarial pass):

- **A 2-frame velocity is below the noise floor.** Centroids are integer px; at
  ~20px/s the true motion over 50ms is <1.1px, so a ±1px jitter aliases to
  ±20px/s. Fix: **Theil-Sen** slope over ≥3 samples spanning ≥0.12s, refusing to
  lead when consecutive samples reverse direction (a turn / noise) or |v|<8px/s.
- **A fixed lead-TIME is wrong** — the hold is closed-loop (varies with charge),
  so the model (charge→distance) carries no time info. Fix: log the measured
  `hold_ms` (was always NULL) toward fitting `time≈f(charge)`; until then use the
  `LEAD_TIME_FALLBACK_S=1.3` constant, on the lead arm only.

Safeguards: the lead is **turn-capped to ±12px** (the slide is monotonic only
~0.8s but the lead spans ~1.3s, so a reversal can fall inside the flight —
capping bounds an over-lead to ~one catch radius), **reach-clamped** (the model
would silently swallow an out-of-reach lead — flagged via `lead_clamped`), and
**mine-rechecked** at the led landing (leading moves it off the fish's cell, so
"aimed at a fish = safe" no longer holds). It operates on the scalar distance
from the fixed origin, so −velocity correctly *decreases* distance (no abs-flip).

**It's an experiment, not a switch** (the effective catch radius is ~10px and the
cast model's own error is heavy-tailed, so a few-px lead error could be
net-negative). Casts alternate `aim_mode` `model_lead` / `model_nolead` by
parity; both arms sample velocity (matched latency) and log
`arm_fish_vx_px_s, lead_time_s, lead_px_intended, lead_px_effective,
lead_clamped, lead_n_samples` + `hold_ms`. Promote off the A/B only on a
disjoint-CI make-rate win under matched velocity bins (the #23/#38 discipline):
filter `aim_mode='model_lead' AND lead_px_effective!=0` vs `model_nolead`,
stratify by `arm_fish_vx_px_s`, compare `made` with Wilson CIs. v1 is single-fish
only (~91% of casts); multi-fish keeps the un-led aim. Pure helpers are
unit-tested (`tests/test_fishing_lead.py`); the live velocity sampling is manual.

## Sources

- IdleOn Wiki — Fishing Minigame: https://idleon.wiki/wiki/Fishing_Minigame
  (sprites: MGgreenfish/MGyellowfish/MGpurplefish/MGbluefish/MGredfish,
  MGfsh5 mine, MGlure)
- IdleOn Wiki — Fishing (skill), "Fishing Minigame" section: https://idleon.wiki/wiki/Fishing
