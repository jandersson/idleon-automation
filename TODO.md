# TODO

## Now

- **Chopping: validate fixes** — 2 attempts left for the day. Run with the
  current settings (`COOLDOWN_AFTER_CLICK=0.45`, `STAGNATION_LIMIT=5`,
  `RED_SAFETY_MARGIN_PX=8`) and see whether it scores cleanly without ending
  on a red-zone misclick. If it still misses at edges, raise the margin or
  add velocity-based prediction.

## Next

- **Chopping: tune red safety margin** if the current 8px is wrong — too
  small means edge misses, too large means the bot waits forever and never
  clicks. Consider deriving from observed leaf velocity instead of a fixed
  pixel count.

- **Chopping: bot starts the game itself.** Today the user manually clicks
  the START button before launching the bot. The bot could click it itself
  by reusing `chopping-pick-button-region` for a separate `start_button`
  region (or repurposing the existing button picker with a second slot).
  Then the bot polls for a "ready" state, clicks start, plays through, and
  exits — fully unattended.

- **Chopping: detect greyed-out START button = no attempts left.** The
  game shows daily attempt exhaustion by greying out the start button (no
  numeric counter shown). Before clicking start, sample the button region's
  HSV saturation; if it's near zero, exit with a "no attempts left today"
  message instead of clicking. Required for an unattended launcher.

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
