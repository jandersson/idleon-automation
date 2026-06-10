# Choppin minigame — mechanics research (2026-06-11)

Community-sourced mechanics model for the choppin minigame, gathered
because attempts are capped (5/day) and each one needs to count.
Cross-referenced against the bot's own first instrumented session
(2026-06-11 00:13, see TODO.md). Sources at the bottom; per-claim
confidence noted since none of this is from the game's code.

## Scoring

- **Green chop = +1 point, yellow/gold chop = +2.** (wiki + DigitalTQ,
  consistent everywhere; high confidence)
- Score thresholds that matter beyond per-attempt rewards: a quest
  wants 150, an achievement at 141 pays 25 gems + 4hr time candy, and
  **mage skill progression scales with the account's best score** —
  so peak score matters, not just average rewards. (Steam threads +
  DigitalTQ; medium-high confidence)
- Rewards per attempt: Choppin EXP, logs of that tree, leaves —
  scaling with points. **Needs ≥5% choppin efficiency on the
  character or the attempt pays nothing.** (wiki summary; medium
  confidence on the 5% number)

## Speed model

- **Edge bounces speed the leaf up** ("Each time it hits an edge it
  speeds up" — Steam tips thread). Time spent not scoring is
  therefore actively harmful: the leaf keeps bouncing and ramping.
- **Yellow chops slow the leaf down** ("Hitting yellow slows down the
  speed the mark moves") — yellow is doubly valuable: +2 AND a brake.
- DigitalTQ instead claims "for each successful chop, the speed
  increases" — conflicts with the Steam model. UNRESOLVED; our own
  polls data can settle it: leaf_vx_px_s over t_ms vs chop times
  (fired=1) vs bounce times (vx sign flips) from any session.
- Bot's measured speeds 2026-06-11: 257–386 px/s within the first
  ~1.2s (3 chops, ~2 bounces deep).

## Round end

- **Clicking while the leaf is over red ends the round.** One
  mistake, no lives. (Steam "broken" thread + our chop #3 death:
  fired 19px ahead of red at ~50–75ms time-to-red, leaf vanished
  250ms later, bar gone.) No timer or chop-cap mentioned anywhere —
  rounds appear to end only by red-click. UNVERIFIED: whether
  inactivity ends a round, and whether a voluntary exit keeps the
  score.

## Zone dynamics

- **A successful chop shifts/re-rolls the zone layout under the
  leaf** ("double clicking moves it twice and you run the risk of
  clicking the red space that moves under the cursor" — Steam tips).
  Matches the bot's RLE observation: red grew r50→r53 / r45→r48 and
  green shrank 120→114 across three chops. Implication: any cached
  layout is stale the instant a chop registers — always re-sample
  before the next click (the loop already does).
- Gold zones exist in some layouts (seen once in a transition frame);
  unknown what makes them appear.

## Strategy (community consensus + implications for the bot)

1. **Front-load chops.** "Click multiple times when the arrow passes
   over the green at the start" is the community max-score line: the
   first seconds are the slowest and safest, and points-per-pass is
   highest before the speed ramp. For the bot this means the
   ~210ms+ inter-chop delay (COOLDOWN_AFTER_CLICK=0.15 +
   random_delay 20–60ms) is the binding constraint early — at
   ~257 px/s the leaf spends ~450ms crossing a 115px green, so only
   ~2 chops per pass fit today. Tuning the cooldown down (with the
   time-to-red gate and post-chop re-sample as the safety net) is
   the highest-leverage change. Open question: the minimum delay at
   which consecutive chops still register in-game.
2. **Always take yellow.** +2 and a slowdown. Worth biasing toward:
   if the leaf is heading toward a yellow zone within the safe
   horizon, prefer waiting for it over a marginal green chop;
   never skip a safe yellow.
3. **Never click red** — the bot's one death condition. The
   directional time-to-red gate (MIN_TIME_TO_RED_MS) is the
   load-bearing control; late-round it will (correctly) starve fires
   as speed rises, at which point the attempt has reached its
   natural ceiling.
4. Edge bounces can't be prevented (leaf motion is autonomous), so
   "avoid the sides" translates for a bot into: score as much as
   possible per unit time early, and bank yellow slowdowns.

## Open unknowns (answerable from our own data / next sessions)

- Chop-vs-bounce speed attribution (polls have everything needed).
- Minimum inter-chop delay that still registers (probe by lowering
  COOLDOWN_AFTER_CLICK stepwise across attempts and checking score
  increments per chop in the session log vs in-game score).
- Does inactivity end a round? Does a voluntary exit keep the score?
  (User observation needed once, during a dead-end round.)
- What spawns gold zones.

## Sources

- IdleOn wiki — Choppin: <https://idleon.wiki/wiki/Choppin> (page is
  Cloudflare-gated to scripts; content via search summaries)
- DigitalTQ Choppin guide: <https://www.digitaltq.com/wiki/idleon/choppin-guide>
- Steam: "Tips for minigames": <https://steamcommunity.com/app/1476970/discussions/0/3807284068743364838/>
- Steam: "The chopping and mining mini-game are not fun or completely
  broken": <https://steamcommunity.com/app/1476970/discussions/0/3167694551647775610/>
