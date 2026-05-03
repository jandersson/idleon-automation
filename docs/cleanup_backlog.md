# Cleanup backlog

Code that was added during investigation/experimentation and is now safe to
delete. Items here exist to keep them tracked and not forgotten.

## Hoops: click-position machinery

**Conclusion from experiments (2025-05-03):** click position has no
measurable effect on ball trajectory.

- click_y sweep (170 → 490, 320px range) → ball_x_at_rim variation: 24px
  (within the ~20px noise floor at fixed click).
- click_extreme (window corners + center vs at-hoop) — ran the experiment
  and didn't see meaningful trajectory differences either.

The earlier "click matters" reading was confounded by perturbation-sweep
luck and a small number of flukey makes. The clean isolation experiments
disproved it.

### Remove

- [ ] `CLICK_STRATEGY` constants and all branches in `_pick_click_position`
      (`varied`, `sweep_y`, `extreme`, `center`) → just always click at
      window center, or wherever; pick the simplest single thing.
- [ ] `VARIED_CLICK_POSITIONS`, `CLICK_SWEEP_Y_OFFSETS`,
      `CLICK_EXTREME_POSITIONS` constants
- [ ] `EXPERIMENT_MODE` env-var dispatch in `minigames/hoops/main.py`
      (keep the launcher's `experiments` config slot, but drop the two
      click experiments from the hoops minigame entry)
- [ ] `tests/test_click_strategy.py` — delete entirely once the picker
      is collapsed to a constant
- [ ] DB rows logged during click-sweep and click-extreme sessions:
      `WHERE session_started IN (...)` — they're polluted relative to
      "normal" play. Probably fine to leave them since the OCR+trajectory
      validation now blocks the false makes from those runs anyway.

### Keep

- `click_x` / `click_y` columns in `shots.db` — still useful per-shot
  diagnostic info even if we don't vary them.
- Launcher's general `experiments` config slot — useful framework for
  future tests (just not these specific ones).
