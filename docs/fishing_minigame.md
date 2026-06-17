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

## Next session: aim off the charge bar (2026-06-17, after 7 live runs)

State: the bot auto-starts (clicks PLAY GAME), detects the green fish cleanly
(#63), casts open-loop via `hold_ms`, and measures the landing by the bobber's
MAX x over a poll (`_measure_landing` — the bobber reels in after landing, so
max-x is the landing, not the last read). Landings are now reproducible, but it
barely scores: the bobber-distance model **undershoots** near a distance **cap
(~267px)** where the charge bar fills up, plus far casts land off-crop (noise).

Key finding — the **red charge bar** (left edge) is a far better signal than
the bobber: `find_charge_level` (red fill height, x<10) maps cleanly and
**reproducibly** to landing distance (36→181, 59→294 both exact; ~4.9
px-distance per px-fill), and it's always in-crop + stable after release.
`casts.charge_level` is now logged every cast (even when the bobber lands
off-crop). Anomaly to explain: ~half of run 7's casts logged `charge_level=0`
AND no bobber — likely cast too soon after the previous (reel-in not done →
no charge); check `CAST_COOLDOWN_S` / sequencing.

Plan:
1. From a fresh run's `fishing.db`, fit `charge_level → landed_dist` (clean) and
   check `hold_ms → charge_level` (is open-loop hold deterministic, or does the
   bar overshoot/oscillate?). Investigate the `charge_level=0` casts.
2. Switch the cast feedback to the charge bar: measure distance from
   `charge_level` (robust, in-crop) instead of the off-crop bobber, so the cast
   model fits on clean data. If `hold_ms → charge_level` is too noisy, go
   closed-loop (charge while polling the bar, release at the target level — a
   new charge-and-release primitive, vs the open-loop `hold()`).
3. Model the distance CAP (linear-up-to-cap, then flat) so aiming doesn't
   over-hold for far fish.

## Sources

- IdleOn Wiki — Fishing Minigame: https://idleon.wiki/wiki/Fishing_Minigame
  (sprites: MGgreenfish/MGyellowfish/MGpurplefish/MGbluefish/MGredfish,
  MGfsh5 mine, MGlure)
- IdleOn Wiki — Fishing (skill), "Fishing Minigame" section: https://idleon.wiki/wiki/Fishing
