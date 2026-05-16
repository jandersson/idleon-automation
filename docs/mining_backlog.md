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

## Cart behavior (issue #1, resolved 2026-05-16)

**Game-logic: the cart auto-scrolls forward immediately after Start.**
**Screen-coordinates: the cart sprite is fixed; the track scrolls past
it from right to left.** Classic endless-runner camera — same motion,
two reference frames.

Evidence: `captures/trace_20260515_220235` (bot clicks Play Game at
T=0, then captures continuously with zero further input).

- Cart sprite pixels at y≈295-330, x≈290-360 are **byte-identical**
  between t=0.0s and t=2.0s — `cv2.absdiff` max=0, mean=0.0 in that
  box. The cart does not move 1 pixel on screen.
- A pit/gap visible at the right edge of the track in frame 0 slides
  leftward and is directly under the cart by frame 28 (t=2.8s) — the
  track moves, not the cart. See `analysis/show_frame_{000,010,020,
  025,028,030}.png`.
- Frame 30 (t=3.0s): cart is gone (fell into pit). Frame 144 (t=14.4s):
  back at Play Game prompt with counter 5→4 (attempt consumed).

Implications for downstream issues:

- **#3 find_cart** is NOT a trivial region lookup (corrected 2026-05-16).
  The minigame overlay floats above the player character's world
  position, so cart screen-x varies between captures —
  `trace_20260515_220235` had the cart at x_frac=0.52,
  `trace_20260516_131332` at x_frac=0.146. Needs dynamic detection.
  Approach: template-match the distinctive cart wheels, or HSV-cluster
  the cart body where it sits ON the plank. Blocks #4.
- **#4 find_next_terrain** scanners are implemented (`_scan_plank_pits`,
  `_scan_plank_ore` in `detector.py`) and validated visually against
  both traces. The integration that picks the "next obstacle ahead of
  cart" is blocked on #3 — without dynamic cart x we false-positive on
  the cart itself.
- **#5 click policy** has a hard deadline per obstacle equal to its
  distance from the cart divided by scroll speed. Pit-tracking on
  `trace_20260515_220235` gives scroll speed ~93 px/s leftward.

## Ore visual signature (verified 2026-05-16)

Ore is **copper/orange-brown chunks protruding UP from the plank
surface**, not the silver-blue described in the wiki/early backlog.
HSV signature is similar to the plank itself (H≈10-14, S>=80, V>=80)
but ore appears in the y-band ABOVE the plank top (normally cave-dark)
and is contiguous with the plank's bright top row. Slamming on ore
rebounds the cart upward (free jump) and scores points.

Captured ore frames in `trace_20260516_131332` around frames 35-50.

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
