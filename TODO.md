# TODO

## Now

- **Chopping: validate fixes** — 2 attempts left for the day. Run with the
  current settings (`COOLDOWN_AFTER_CLICK=0.45`, `STAGNATION_LIMIT=5`,
  `RED_SAFETY_MARGIN_PX=8`) and see whether it scores cleanly without ending
  on a red-zone misclick. If it still misses at edges, raise the margin or
  add velocity-based prediction.

- **Hoops: tune OFFSET_ANCHORS_DIRECT for 960x572 window.** Current entries:
  `(400, 50)`, `(700, 50)`, `(835, 14)`, `(900, 11)`. The first anchor was a
  starting guess; in the latest session 7 shots at hoop (606, 416) all
  missed into the backboard with offset=50 (target_y=466). Need real shots
  to bias-correct. Suggested method: keep the bot running for one full
  attempt and watch which side of the rim shots land (over/under), then
  nudge the (400, …) anchor by 5-10px in the right direction. Repeat until
  the bot makes consistently in this hoop_y range.

## Next

- **Chopping: tune red safety margin** if the current 8px is wrong — too
  small means edge misses, too large means the bot waits forever and never
  clicks. Consider deriving from observed leaf velocity instead of a fixed
  pixel count.

- **Hoops: detect swish (+2) vs direct make (+1).** The `score_changed`
  diff only knows "digits changed", not by how much. Last session a 0→2
  swish was logged as a single +1 make. To tally accurately, read the
  actual digit value (template-match each digit, or OCR the score crop)
  and use the delta. Affects session score reporting and trophy
  progress — not load-bearing for whether the bot fires.

- **Hoops: verify lives region.** Same regions.json had `lives` at
  top_frac=0.28 — likely also pointing at empty sky in the 960x572
  window. The lives counter is only used as a diagnostic (`[lives]
  counter ticked down`), not for control flow, so this doesn't break
  anything but the diagnostic is noise. Re-pick when next in front of
  the game.

## Someday

- **Tighten `common.window.get_bounds`** so it doesn't match the first window
  containing the substring. Today it works because every minigame uses the
  unique "Legends Of Idleon" title, but if a future window title contains
  that string the matcher silently picks the wrong one. Could prefer exact
  match, or filter by process name.

## Done

- Use full "Legends Of Idleon" window title across all four minigames
  (`bf155f2`..`faa3678`).
- Chopping: stagnation guard + cooldown bump (`6ec43c8`).
- Chopping: red safety margin to avoid edge misclicks (`d356d6b`).
- Hoops: never fire clamped shots; drop game-start-click loophole (`3c7f7e6`).
- Hoops: drop top OFFSET_ANCHORS_DIRECT anchor 90→50 for current window
  (`ff69fdc`).
- Hoops: fix score region — was pointing at empty sky (`3fccfea`).
- Tkinter launcher under `ui/` with one row per minigame (`3df1ab6`).
