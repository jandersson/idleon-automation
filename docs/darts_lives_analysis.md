# Darts: lives & the red-chain +1-life mechanic (#56)

*Analysis 2026-06-17 over `minigames/darts/assets/darts.db` (993 throws, 56
sessions). Cross-checked by an independent re-derivation; the fragility
caveats below are the corrected, honest version.*

## The mechanic (wiki-confirmed)

From idleon.wiki (*Minigames* → Throwy Darts), verbatim:

- "The player starts with **3 lives**, and loses 1 for missing the wall."
  A miss = the dart not hitting a colored section: missing the wall,
  hitting a divider, or hitting a **deactivated** stripe.
- Stripe values: **Red (bullseye) +5, Green +3, Tan +2, Gray +1.**
- Deactivation (cumulative-hit gated, already modelled in #51): **Gray
  dead after the 10th hit, Tan after the 25th.** Green/Red never die.
- **"Red | +5 | 3 consecutive hits grant +1 life"** — three reds in a row
  bank one extra life. No max-life cap stated.
- Separately: **9 bullseyes in a row** awards the *Nine Dart Finish*
  trophy (a one-time reward, not a life mechanic).

So the life economy is: start 3, −1 per miss, +1 per 3-red chain.

## What the data shows

**1. The +1-life chain is effectively dormant.** Across all 993 logged
throws there is **exactly one** run of 3 consecutive bullseyes (ids
346–348, session 2026-06-10, throws 15–17). The longest consecutive-red
run in the entire dataset is 3. Per-game max-red-run distribution
(58 reconstructed games): `0→13, 1→23, 2→21, 3→1`. The bot banks a life
about once per thousand throws — the mechanic is essentially never
exercised under the current aim.

**2. Bullseyes are if anything slightly anti-clustered.** Bullseye rate
among hits is **20.8%** (173/833; 17.4% of all throws). Under
independence you'd expect ~3.9 three-in-a-row events across this dataset
(Monte-Carlo over the actual game lengths at p=0.174); we observe **1**.
That gap is only weakly significant — P(≤1 under independence) ≈ 0.10
(~1.6σ) — and a direct serial test agrees in direction but not strength:
P(bull | prev bull) = 0.156 vs P(bull | prev non-bull) = 0.180. The
likely cause is the wiki's "**the platform and player are relocated at
random after every hit**": each red lands you in a new spawn, so the vy
band that just produced a red doesn't transfer to the next throw. Read
this as *suggestive, not established* — the bot doesn't try to chain, so
there's little signal either way.

**3. The DB cannot measure whether runs are life-gated.** This is the
load-bearing caveat. A natural analysis — "do games that hit a 3-red
chain last longer?" — is **not answerable from the current schema**:

- **In-game score is monotonic within a session.** Of 58 reconstructed
  game boundaries, only **2** came from an actual within-session score
  reset; the other 55 are just session changes. The score never drops to
  delimit one game from the next, so game boundaries are invisible to any
  score-based reconstruction. In practice a darts session ≈ **one game**
  (the bot plays one game and the session ends at game-over detection),
  which is why "one game per session" reproduces the data.
- **Free startup misses are conflated with costly game misses.** Darts
  has an unlimited startup phase (you must land one throw to begin; misses
  there cost no life), then the 3-life game. Both log identically as
  `hit=0`. So "misses per game" overcounts lives lost, and the modal
  "3 misses per game" is *consistent with* the 3-lives model but is **not
  independent evidence** for it.

The upshot: the DB can confirm the chain is dormant, but it **cannot**
confirm a +1-life event fired, cannot count lives, and cannot measure the
value of banking one. Issue #56's "next session" hope — measure the
life-gating effect from existing columns first — isn't achievable from the
current schema. **Visual life detection is a hard prerequisite, not an
optional follow-up.**

## Implication for #56 — disciplined path

1. **Red-streak instrumentation [done 2026-06-17].** `shot_stats["red_streak"]`
   counts consecutive bullseyes (resets on any non-red outcome or an
   unreadable score), logged to the new `throws.red_streak` column and
   printed live when it reaches a multiple of 3. Makes future chains
   directly queryable instead of reconstructed, and is the prerequisite
   #56 step 2 asked for. No aim change.
2. **Visual life/heart detection (next).** Find the heart-icon signal on
   the scoreboard (candidates in `assets/diagnostics/`), template-match it
   per the dynamic-UI convention, and log a `lives` column per throw. Only
   this gives game boundaries and a way to confirm a +1-life event.
3. **Only then** measure: do 3-red chains actually extend runs (more
   throws → more points), and is biasing aim toward red when `red_streak`
   is 1–2 deep net-positive? It may not be: the E[stripe] model already
   favours red (+5 is the max stripe), the ~21% bullseye rate makes
   reliable 3-chaining hard, and high-variance bullseye aim risks *more*
   misses (each a lost life). Compare run length / total score with vs
   without, Wilson CIs, #23/#38 discipline. Don't build the aim change
   before a measured effect.

The honest current read: chasing the life mechanic is a real bet, not a
freebie, and it's unmeasurable until life detection lands.
