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

- **Leaf motion is EASED, not constant-speed** (measured, 00:46
  full-rate session): mid-bar ~630 px/s vs ~290–350 near the edges,
  with sample density to match — sinusoidal-ish, like the hoops
  platform bob. Gate consequence: an instantaneous vx sampled in the
  slow edge region understates the speed the leaf reaches crossing
  toward mid-bar red; the gate floors its projected speed at half the
  recent-max |vx| (EASING_SPEED_FLOOR_FRAC).
- **No measurable per-chop or per-bounce ramp at low scores**
  (measured, same session): peak speed per sweep 786/580/791/724 px/s
  across 0→5 chops and 3 bounces — flat. The maintainer's per-chop
  intuition and the Steam per-bounce claim both remain possible at
  higher scores (the "hyperspeed" reports come from much longer
  rounds); a longer round's polls will tell.
- **Yellow chops slow the leaf down** ("Hitting yellow slows down the
  speed the mark moves") — yellow is doubly valuable: +2 AND a brake.
  (Community claim, not yet observed — no yellow chopped yet.)

## Chop registration (measured, 00:46 session)

- A registered chop re-rolls the zone layout within **86–201ms** —
  the re-roll is the in-game ack.
- The re-roll is NOT the end of the game's chop cooldown: clicks
  198/201/225ms after a registered chop were **silently ignored** (no
  re-roll, no point), while 655/720/812ms gaps registered. The
  registration boundary sits somewhere in (225, 655)ms;
  MIN_INTERCHOP_S=0.45 is the current bisection probe and each
  session's polls tighten the bound for free.
- **Ignored clicks are not free**: the 00:46 round's fatal click came
  225ms after its predecessor — no point was possible, but the death
  still happened. Working model: every click is evaluated against the
  logic-leaf position (green during cooldown → nothing; red → death,
  always). So clicks during the cooldown window carry pure downside —
  the fire hold now enforces the full interval.
- The fatal click itself: the leaf read the same x twice 127ms apart
  (stale/frozen render or capture) → vx = exactly 0.0 → the old
  `abs(vx) > 1e-6` check disarmed the directional gate → click landed
  with the real leaf likely ~355 px/s and red ~31px (~87ms) ahead.
  Direction-unknown fires are now banned (MIN_VX_FOR_FIRE).

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
   highest before the speed ramp. IMPLEMENTED 2026-06-11: the fixed
   150ms post-chop sleep became a layout-settle hold — the bot
   re-arms the moment the zone layout re-rolls (the in-game signal
   the chop registered), with 150ms as the fallback deadline only.
   The polls record the actual settle lag (fired=1 row → first
   changed zone_layout), which calibrates whether the fallback can
   shrink further.
2. **Always take yellow.** +2 and a slowdown. IMPLEMENTED 2026-06-11
   as the same-sweep gold upgrade: a safe green fire is deferred when
   gold lies ahead of the leaf before any red (same sweep, no extra
   bounce, capped at GOLD_RIDE_MAX_MS); a leaf already over gold
   fires as before. Cross-pass gold waiting was deliberately NOT
   built — its value depends on the unresolved speed-attribution
   question below.
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
