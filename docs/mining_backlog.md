# Mining backlog

Open work for the mining minigame, captured at the end of the
scaffolding session (2026-05-09). The folder, entry points, and
launcher row are in place; the bot itself does nothing useful yet
because the detectors are stubs and the play region won't survive a
position change.

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

A reference frame sits in `minigames/mining/assets/captures/` from the
scaffolding session (timestamp `20260509_023207_*.png`). Note the
captured run shows the game in its pre-roll state — score is "0 PTS"
across all 60 frames and the cart never moved, suggesting the cart
auto-scrolls only after the first input. Re-capture during an actively
clicked attempt before designing motion-dependent logic.

## Open work

### 1. Anchor-template localization (blocks everything else)

The picked play region in `assets/regions.json` is window-relative
fractions, but the minigame overlay floats with player position. A
fixed fractional region will break the moment the player stands
somewhere new.

Replace the static region with a runtime anchor lookup:
- Crop a tight, distinctive template from a real frame — candidates
  are the score-text "PTS" or "BEST" labels (stable, contrasty), or a
  unique corner of the panel border.
- On each tick, `matchTemplate` over the full window to locate the
  anchor, then derive the play region as `anchor_xy + fixed_offset`.
- The pattern mirrors how `hoops` finds the rim — apply it to the
  minigame's own UI chrome instead.

### 2. find_cart

`detector.py:find_cart` returns `None`. Needs to return the cart's
(x, y) in the play region. Cart is small (~20×15 px), reddish-brown
body, dark wheels. Two reasonable approaches:
- Template match against a cropped cart sprite (capture both grounded
  and airborne variants — the body might recolor or the wheels might
  disappear when the cart leaves the rail).
- HSV mask on the cart's reddish body color, then bounding-box the
  largest connected component within the play region.

Cart's X may be roughly fixed (track scrolls past), in which case we
mostly care about Y to know grounded vs airborne. Verify against real
in-motion frames.

### 3. find_next_terrain

Returns `None`. Needs to look ahead of the cart and classify the next
terrain feature as ore / pit / trap, plus return the lookahead distance
in pixels. Probably HSV: ore has a distinctive silver-blue palette,
pits are dark voids in the track. Traps weren't seen in the scaffold
captures — confirm what they look like.

### 4. Click policy

Once `find_cart` and `find_next_terrain` are real, `main.py:_run_inner`
needs the actual jump/slam decision. Rough sketch:
- If next feature is ore at distance D, fire jump when D matches the
  cart's jump-arc apex; fire slam when cart is over the ore.
- If next feature is pit at distance D, fire jump just before D == 0
  to clear it; don't slam during the pit.
- The CLAUDE.md "click timing" rule applies: fire the click first,
  then save monitoring frames, log, etc. Sampled world state ages by
  every ms of latency between sample and click.

### 5. Game-over / score detection

Parity with hoops: detect when the run ends (likely a results overlay)
and read out final score for shot-log purposes. OCR the "PTS" digits
during play for live progress, or capture the post-run score screen.

### 6. Dailies counter

Per-day attempt usage shares a counter with chopping and catching
(`common/tries_counter.py`). Wire it once mining is firing real clicks.

## Pointers

- Wiki overview: <https://idleon.wiki/wiki/Mining>
- DigitalTQ guide:
  <https://www.digitaltq.com/wiki/idleon/mining-guide>
- Capture/observe scaffolding: see `minigames/mining/main.py`,
  `capture.py`, `observe.py`, `pick_play_region.py`.
- Reference frames: `minigames/mining/assets/captures/` (gitignored —
  re-run `mining-capture` if you need them).
