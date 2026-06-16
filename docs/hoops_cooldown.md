# Hoops (and darts) minigame cooldown formula — #22

Derived 2026-06-14 from 530 save observations (`minigames/hoops/assets/observations/cooldown.jsonl`, spanning 2026-05-24..06-14) plus an online sourcing pass and a darts cross-check. Referenced from `common/idleon_save.py` (`minigame_cooldown_ticks_for_plays`).

## Bottom line

After the **Nth play of the day**, the game sets the minigame reset timer to:

```
cooldown_ticks(n) = min(900, 12·n² + 66·n − 26)        # provisional constants
cooldown_seconds  = cooldown_ticks / 5
```

- **Driver:** `OLA[424]` = "Hoops Played Today" (the repo's "streak" column). Resets at the daily reset, +1 per play.
- **Cap:** ~900 ticks = **180 s = 3 minutes**, reached around the 7th play.
- **Account-wide and daily-reset** — matches the user's anecdotal model.
- **Computed, not stored:** no OLA index holds a per-play "base". Darts uses the *same* computed curve even though its `OLA[442]` is flat — see below.

## Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Escalates **quadratically** with plays-today | High | 530 obs, 2 clean days, darts cross-check |
| Driver is `OLA[424]` plays-today | High | corr(set, streak)=+0.87; community schema; resets daily |
| Cap ≈ 900 ticks = 180 s (3 min) | High | raw max observed cd = 893t; extrapolated peaks 896–898t |
| Exact polynomial constants | **Medium** | observations are lower bounds (sampled ~1 s post-flush) |

The exact constants are not pinned because every fresh read is caught ~1 s after the save flush, by which point the timer has already decayed a few ticks — so observed values are lower bounds. Candidate fits that all match the rising region within noise:

- `min(900, 12·n² + 66·n − 26)` — best fit to the per-play medians (RMS ~1.4 s), **used in code**.
- `min(900, 25·n²)` — cleanest round form (caps at n=6).
- `min(900, 13·n² + 55·n)`.

Per-play medians observed (ticks): n1≈47, n2≈165, n3≈278, n4≈418, n5≈608, n6≈797, then plateau ~840–898.

## The "15 minutes" correction

The anecdotal cap of 15 minutes is a **ticks-vs-seconds confusion**: the save stores the timer in *ticks* at **5 ticks/second**. 900 ticks reads like "900" (→ 15 min if mistaken for seconds) but is **180 s**. Nothing in the data approaches 4500 ticks (a true 15 min).

## Two corrected OLA labels (were wrong in the repo)

Verified against the authoritative community index map (MrJoiny/Idleon-Injector `optionsAccountSchema.json`):

- `OLA[442]` — **darts high score**, *not* a darts cooldown base. Its flat 125 is a high score that happened to be static; the repo's old `_OLA_DARTS_COOLDOWN_BASE` name was wrong (corrected; `read_darts_cooldown` no longer returns a bogus `base_cooldown_seconds`).
- `OLA[434]` — **darts minigame points** (a lifetime accumulator: 153→2089 monotonic across the whole span, never resets daily). It is *not* a hoops base — the observer logged it as a base candidate; that's now noted as corrected.

The darts cross-check is what settles "computed vs stored": darts' stored `OLA[442]` is flat 125, yet darts' set cooldown follows the *same* quadratic-by-plays curve as hoops. So both games compute `min(cap, quadratic(plays))` from the play count.

## How to finalise the constants

One clean day of capture removes the lower-bound bias:

1. After the daily reset, play hoops back-to-back for plays 1–7.
2. Poll the save aggressively right after each play; record `OLA[423]` at `save_age < 0.5 s`.
3. Fit `cooldown_ticks` vs play index — expect something near `12n²+66n−26` (or a clean form like `25n²`), and confirm the cap is exactly 900 vs ~898.

Alternatively, extract the game's `z.js` bundle and read the `ActorEvents_510` reset handler that writes `OptionsListAccount[423]` from `[424]` — that yields the literal expression.

## Not published online

The escalating-per-play *mechanic* is documented (FearLess Cheat Engine, the wiki: "makes successive reset timer longer"), but the numeric formula and cap are not published anywhere searchable — they were derived empirically here.

## Daily-reset time-of-day estimate

Separate from the per-play cooldown *duration* above: `OLA[424]` (and the darts plays counter) zero out at the account's **daily reset**, a fixed wall-clock time that is **per-account and Pocketwatch-adjustable** (the wiki *Silver Pocketwatch* / Consumables: "Hold down to change your daily reset time by 15 minutes"). So there's no universal reset time — it's derived per account from the logged observations.

`common.hoops_cooldown_observer.estimate_daily_reset()` recovers it: a daily reset shows as a consecutive-flush pair where `streak` or `darts_plays_today` drops to ≤ 1 (a mid-day decrement of `streak` to a higher value is decay, not a reset, and is excluded). The reset fires somewhere inside the gap between the last pre-reset flush and the first post-reset flush, so only **tight** boundaries (gap < `RESET_TIGHT_GAP_S` = 900 s) localise it; their midpoints are averaged with a **circular mean** over time-of-day (so values either side of midnight don't cancel). The launcher's tries strip shows `🔄 Reset: ~HH:MM (in …)` from this, with a `?` marker when confidence is low (one sample, or wide spread).

For the current account the estimate is **≈ 01:52 local (CEST) ≈ 23:52 UTC**, stable across 2026-06-10..16 (spread 0.3 min). Note `OLA[424]` stays 0 from the reset until the first play of the day — so the clean plays-1..7 capture for the constants just needs to be the day's *first* hoops session, not literally run at 01:52.
