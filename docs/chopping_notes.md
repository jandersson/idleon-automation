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

## Speed model (settled 2026-06-11, 01:08 session — 97s, 19 chops)

- **Chops ramp the speed; bounces do NOT.** Mean speed doubled during
  the 9-chop opening burst (308 → ~480 px/s in 10s), then stayed flat
  ~475–590 from t=10s to t=90s across ~a hundred chop-free bounces
  (the gate-starve droughts froze the chop count — a natural
  controlled experiment). The maintainer's per-chop intuition was
  right; the Steam "edges speed it up" claim is refuted at this score
  range. Ramp ≈ +2–5% per chop, apparently saturating-ish.
- **Therefore waiting is FREE.** A skipped fire window costs
  wall-clock, not points or difficulty. Deaths are the only real cost
  in this game. The front-loading doctrine ("time not scoring is
  actively harmful") is WRONG and is hereby retracted.
- **No inactivity timeout** at least up to 45s (the longest drought —
  the round did not end). The 97s round ended only by the bot's own
  marginal fire.
- **Leaf motion is EASED**: mid-bar median ~550–750 px/s vs ~280–420
  near the edges (profile is trapezoid-ish — broad fast middle, slow
  edges). Gate consequence: instantaneous vx near the edges flatters
  time-to-red; the gate floors projected speed at
  EASING_SPEED_FLOOR_FRAC of the recent max (0.75 since the 01:08
  death — chop 20 fired at nominal ttr=150ms on vx=425 while the true
  crossing speed was ~630-750, real ttr ~100ms).
- **True-speed kill boundary ≈ 100–115ms**: all three deaths to date
  had true ttr ≤ ~110ms; survivals observed at 115–123ms.
- **Yellow chops slow the leaf down** ("Hitting yellow slows down the
  speed the mark moves") — community claim, still unconfirmed: 7 gold
  chops in the 01:08 session produced no visible dip (any brake may be
  cancelled by the chop's own ramp).

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

## Open unknowns

- ~~Chop-vs-bounce speed attribution~~ — settled: per-chop (above).
- ~~Does inactivity end a round?~~ — no, at least ≤45s.
- ~~Does a voluntary exit keep the score?~~ — YES (maintainer,
  2026-06-11): points bank as in-game tokens on exit; each game
  starts fresh. Exit-on-starve implemented (STARVE_EXIT_S=60): after
  60s without a safe fire the bot ends the session with the score
  summary and the user exits in-game to bank. Every bot death so far
  came from finally taking a marginal window during a starve — this
  removes that failure mode entirely.
- Minimum inter-chop delay: all 560ms+ gaps registered; the (225,
  560)ms range is untested but moot while the fire cadence is
  crossing-limited anyway.
- What spawns gold zones (first gold appeared at chop 7-8 in the
  01:08 session and persisted; gold width shrank o41 → o33 over the
  round).
- Whether the per-chop ramp keeps compounding at higher scores
  ("hyperspeed" reports) or truly saturates.
- **Next gate lever (designed, not built): position-aware eased
  time-to-red.** The 0.75×recent-max speed floor over-penalizes fires
  launched from the slow edge regions (the leaf genuinely takes
  longer to reach a far red than d/eff_speed says, because it
  accelerates gradually). Model the sweep as x(θ)=(W/2)(1−cosθ),
  v=V_max·sinθ: time-to-red = Δθ/ω with θ=arccos(1−2x/W) and
  ω=2·V_max/W, V_max estimated robustly from the recent window
  (guard the arccos near edges; validate against all recorded fires —
  the three deaths must compute under threshold, the survivals
  over). Opens late-round edge-launched fires without loosening
  mid-bar safety. Lower priority now that exit-on-starve banks
  tokens instead of risking marginal fires.

## Sources

- IdleOn wiki — Choppin: <https://idleon.wiki/wiki/Choppin> (page is
  Cloudflare-gated to scripts; content via search summaries)
- DigitalTQ Choppin guide: <https://www.digitaltq.com/wiki/idleon/choppin-guide>
- Steam: "Tips for minigames": <https://steamcommunity.com/app/1476970/discussions/0/3807284068743364838/>
- Steam: "The chopping and mining mini-game are not fun or completely
  broken": <https://steamcommunity.com/app/1476970/discussions/0/3167694551647775610/>
