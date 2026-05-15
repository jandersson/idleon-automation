# Mining backlog (historical)

Captured at the end of the scaffolding session (2026-05-09). Open work
items have moved to GitHub issues — see
[`label:minigame:mining`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22minigame%3Amining%22).
This file keeps the visual / mechanical context so each issue stays short.

## What the minigame is

Cart-jump-slam, rendered as a small overlay panel **anchored above the
player's head** in whatever world position the player is standing.

- Wooden horizontal track with a small brown mine cart on it.
- Silver-blue ore chunks sit on the track at intervals — landing a slam
  on these scores points.
- Dark gaps in the track are pits — landing on these ends the run.
- Score readout shows "X PTS" (left) and "Y BEST" (right) above the
  track. Player character name floats below the track.
- 5 daily attempts, pooled with chopping/catching.

## Reference frames

`minigames/mining/assets/captures/capture_20260509_023207_*.png` (60
frames, gitignored — re-run `mining-capture` if you need them).

The original 60-frame capture (referenced above) was misread: those
frames were of the overworld with the player standing at the Mining
Helper NPC, not the active minigame. A proper trace was captured on
2026-05-15 via `mining-trace`, which clicks the in-game "Play Game"
button itself and then captures continuously without further clicks.

## Cart behavior (issue #1, resolved 2026-05-15)

**The cart waits for the first jump click. It does NOT auto-scroll.**
If no input arrives, the attempt self-terminates after ~2-3 seconds
and the daily-attempts counter still ticks down. From
`captures/trace_20260515_220235`:

- t=0.0s: bot clicks "Play Game" button
- t=0.0 → t=~2.4s: cart sits at start of track, identical position
  every frame (`analysis/trace_kymograph2.png` shows a vertical streak
  — zero horizontal motion)
- t=~2.5s: cart + track disappear (attempt ends without a single jump)
- t=~5s: UI returns to the Play Game prompt, attempts counter dropped
  5→4

Implication for the click policy (#5): the bot needs an "ignition"
click to start cart motion, and must fire it within the ~2-3s grace
window or the attempt is wasted.

## Tooling notes

- `mining-pick-start-button` saves a region around the in-game "Play
  Game" button. Always grabs a fresh live screenshot — the prompt is
  only visible in a specific game state, so reusing stale captures
  doesn't work.
- `mining-trace --seconds N --startup S` clicks Start, then captures
  N seconds of frames + logs human clicks. Trace dir contains
  `frame_NNNN.png` and (on clean exit) `trace.json` with click
  timestamps tied to frame indices.

## Pointers

- Wiki overview: <https://idleon.wiki/wiki/Mining>
- DigitalTQ guide: <https://www.digitaltq.com/wiki/idleon/mining-guide>
- Scaffolding: `minigames/mining/main.py`, `capture.py`, `observe.py`,
  `pick_play_region.py`.
