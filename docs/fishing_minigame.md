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
for their 2 points. Caveat: one template/pose so far — add crops if other eel
animations under-match.

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

## Targets slide — the moving-target lead (the streak stall, #58, 2026-06-18)

The streak stalls at ~3 (so eels/squid/whale rarely unlock, and the run is all
greens). Cause, measured from the saved botrun frames + `fishing.db`: **the
targets SLIDE back and forth along the bar**, starting ~cast 5 and growing in
speed/amplitude through a game. Within one cast the motion is monotonic/linear
at **~±15–22 px/s** late-game, then it settles at a turn point. The bot aimed at
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
