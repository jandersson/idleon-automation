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

The captured run shows the game in its pre-roll state — score is
"0 PTS" across all 60 frames and the cart never moved, suggesting the
cart auto-scrolls only after the first input. This is the open
question tracked in #1.

## Pointers

- Wiki overview: <https://idleon.wiki/wiki/Mining>
- DigitalTQ guide: <https://www.digitaltq.com/wiki/idleon/mining-guide>
- Scaffolding: `minigames/mining/main.py`, `capture.py`, `observe.py`,
  `pick_play_region.py`.
