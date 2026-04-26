# Hoops strategy notes

What the bot assumes about the game, why each tuning knob exists, and how
to reason about changes to it. Living document — update as we learn more.

## Game-physics assumptions (empirical)

- **Launch is fixed-power and fixed-direction.** Same platform Y at click +
  same hoop position → same trajectory, every time. Tuning is therefore
  deterministic per-shot, not statistical.
- **Hoop template-center ≠ rim opening.** Our `hoop.png` was cropped to
  include the backboard, so the matched-region center sits in the middle of
  the backboard. The actual rim opening is **lower** (larger Y) than the
  matched center. Positive `OFFSET_ANCHORS` values reflect this gap.
- **Hoop respawns randomly between shots** at score 0-19 — only the Y
  varies; X is essentially fixed.
- **At score ≥10**, the platform starts moving horizontally too (we have
  `home_x`/`X_TOLERANCE` machinery to wait for the platform to be at its
  resting X before firing — currently disabled by setting tolerance=9999).
- **At score ≥20**, the hoop also moves horizontally each frame — our
  per-shot hoop position becomes stale by the time the ball arrives. Not
  yet handled.
- **At score ≥30**, the hoop moves both axes. Not handled.

## Scoring

- Direct make = **1 pt**.
- Nothing-but-net (ball through rim without touching rim or backboard) =
  **2 pts**.
- Trophy at single-trial score **40+**.
- Pet from combined score **66** across 3 trials.

## Two switchable strategies

`SHOT_STRATEGY` in `main.py` toggles between two whole approaches.

### "direct" (default)

Tune `OFFSET_ANCHORS_DIRECT` so the ball arc passes through the rim
naturally. Mid-flight rescue runs as a safety net for marginal overshoots
but is not the primary make mechanism — `BALL_X_TOLERANCE=18` is wide
enough not to interfere with shots that would have made on their own.

Empirically known to work (6/7 in an earlier session). Sensitive to small
offset errors (a few pixels too high or low and shots miss).

### "overshoot"



We deliberately tune `OFFSET_ANCHORS` so the launched ball passes **above**
the rim, not through it. The mid-flight rescue (`_try_rescue`) then clicks
on the ball when it crosses the rim's X — per the wiki, that drops the ball
straight down. Result: the drop goes through the rim cleanly = swish = 2 pts.

Why this beats "tune for direct make":
- Single decision variable per shot (rescue trigger), not two (offset + rescue).
- A swish is worth 2x a direct make → faster trophy progress.
- Robust to small tuning errors: as long as the ball is *somewhere over* the
  hoop X when rescue fires, it drops in.
- Direct-make tuning was sensitive: 11 was good, 8 under-arced, 14 over.
  Overshoot tuning is wider — anything that gets the ball ~30+px above the
  rim works, so the offset window is much bigger.

What we need for this strategy to work:
- Reliable ball detection (HSV mask + area filter; tuned via
  `BALL_HSV_LOWER/UPPER` and `BALL_MIN_AREA/MAX_AREA`).
- Tight `BALL_X_TOLERANCE` (currently 12) so the click drops the ball at
  the rim opening, not on the backboard.
- Long enough `RESCUE_WINDOW` (1.5s) to catch the ball during its descent.

## Tuning knobs

| Knob | Purpose | Tune when |
|--|--|--|
| `OFFSET_ANCHORS` | Position-dependent platform launch Y | Misses are systematically over/under at a given hoop Y range |
| `Y_TOLERANCE` | How close platform Y must be to target before firing | Bot fires too often (high) or never (very low) |
| `BALL_X_TOLERANCE` | Rescue trigger window | Drops are off-rim (too wide) or rescue rarely fires (too tight) |
| `BALL_MAX_AREA` | Filter for ball+nearby-pixel merged blobs | Ball detection vanishes near hoop (raise) or false-detects rim (lower) |
| `BALL_HSV_LOWER/UPPER` | Color mask for the ball | Background or character orange is being matched as the ball |
| `RESCUE_WINDOW` | How long to track the ball after launch | Rescue runs out before ball reaches hoop_x (raise) |

## Open problems

1. **Ball detection sometimes loses the ball** during the second half of
   flight (near the rim). Likely cause: HSV mask catches both the ball and
   the orange rim, merged blob exceeds `BALL_MAX_AREA`. Bumped MAX to 2500;
   monitor folders save flight frames for offline debugging.
2. **Score 20+ moving hoop** not handled — bot uses a stale hoop position.
3. **Suspicious clamped MAKEs** with very high score-diff values (47, 18, 13)
   when the shot was geometrically unmakeable. Either the shot really did
   make (lucky bank) or score region picks up a UI animation. Worth
   investigating with monitor screenshots.
