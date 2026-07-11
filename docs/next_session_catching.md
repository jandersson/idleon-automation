# Next session: catching minigame pivot (written 2026-07-11)

Copy-paste prompt for the next Claude Code session. Context: the
2026-07-06/11 chopping sessions took that bot from starving-at-20 to
best-52 (human best 66) via planned shots + measured error models;
chopping is now hands-off and needs only layout luck. Pivoting to
catching.

---

Pivot to the CATCHING minigame in idleon-automation
(C:\Users\jonaS\dev\idleon-automation). Work in a fresh git worktree
per CLAUDE.md (one session = one worktree + branch; merge to main
yourself when done).

READ FIRST, in order:
1. CLAUDE.md (repo root) — cross-cutting patterns. Note the NEW
   "Chopping: established findings" block: the planner-era lessons
   (planned impacts over reactive gates, measured sigma budgets,
   EV ranking, visual overlay auto-location, template OCR, deaths/
   scoring verification via step-function OCR) are the playbook that
   took chopping from 20 to 52 — port what fits.
2. GitHub issues (label minigame:catching): #47 (tracking — early/
   untuned fly bot), #52 (score by hoop colour: orange1/green2/gold3/
   lava10, lava after 20 in a row), #60 (auto-anchor detection to the
   PLAY GAME prompt + biome-specific avatar/hoop colours — chopping's
   find_bar_rect/find_play_button now exist as the canonical
   pattern), #61 (phase-lock bob apex to ring crossings), #62
   (recalibrate sim.py to the real collision model — threads 0 vs
   reality's 12).
3. minigames/catching/{main.py,detector.py,controller.py,
   fit_dynamics.py,catch_log.py} — the bot: Flappy-Bird-style, click
   = flap, navigate hoops. main.py is also its config.
4. minigames/catching/assets/catching.db — 632 flaps / 18 runs
   logged. Query it before asking anything about past runs.
5. docs/chopping_notes.md — as a reference for what a finished
   mechanics+architecture model looks like; mirror the structure for
   catching as findings accumulate.

STATE: early bot with real instrumentation (README row updated
2026-07-11 — the old "Scaffold" claim was stale). Launcher has
Save frames / Trace / Use model toggles (CATCHING_SAVE_FRAMES,
CATCHING_TRACE, CATCHING_USE_MODEL); digit templates 2/10 captured.
The dynamics-fitting path exists (catching-fit-dynamics ->
assets/dynamics.json) but the fitted model is off by default.

SHARED-POOL WARNING: catching attempts draw from the same daily pool
as chopping/mining. Check the launcher header's tries readout (reads
the save) before spending; chopping sessions today may have drained
it.

TRANSFERABLE CHOPPING LESSONS (don't re-derive):
- In-world overlay UI anchors above the player — never cache
  coordinates; auto-locate visually (find_bar_rect pattern) and hang
  companion regions off the found anchor. Auto-click the Play Game
  prompt (shared sprite, minigames/chopping/assets/play_button.png
  matches at 0.99+) for zero-idle starts.
- If a time-based mechanic is suspected (ramps, cooldowns), state it
  to the user as a checkable in-game claim BEFORE shipping behavior
  on it (memory: verify-mechanics-with-user — two same-evening
  theories died on one sentence each).
- Tesseract cannot read Idleon's pixel fonts; template OCR
  (common.score_template_ocr) with per-game digit libraries, harvested
  from --save-frames crops with known values.
- Prediction beats reaction: chopping's reactive gate starved where
  the planner thrives. Catching's flap timing (#61 phase-lock) is the
  same shape of problem — plan impacts/apexes, measure the error
  budget from logged residuals, and rank choices by EV with an
  empirically-calibrated risk curve.
- Instrument first (per-action DB rows + full-rate polls +
  --save-frames), then let post-mortems drive constants. Replay every
  death/failure against the logged state.

WORKFLOW REMINDERS: bots run from the MAIN checkout (launcher `uv run
idleon`, or the Desktop "Idleon Launcher" shortcut); user starts
runs; read logs/DB yourself — never ask for paste-backs. Commit per
logical unit (Conventional Commits), push to main 07:00-23:59
Europe/Stockholm (ask outside the window). Log substantial work to
the Obsidian daily note (## Worklog). File an issue for every
refinement you notice. The chopping-session worktree branch
(claude/modest-feynman-3cc858) is fully merged into main and safe to
remove.

SUGGESTED SHAPE: read the state (1-5) -> query catching.db for what
the 18 runs already show (flap dynamics residuals, death causes,
score distribution) -> fix the cheap stale things (#60's auto-anchor
is nearly free — port chopping's pattern) -> pick between #62
(sim recalibration, desk work) and #61 (phase-lock, needs runs) as
the session's build target -> have the user run 1-2 attempts with
existing tooling before building anything big.
