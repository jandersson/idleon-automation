# macOS port plan + cross-machine data shipping

Status: plan (nothing implemented yet; survey by subagent 2026-06-11).
Target: the repo cloned on a MacBook Air (Apple Silicon); goal (1) bots
runnable on macOS, README platform table flipped to supported once one
bot runs end-to-end; goal (2) the Windows-collected training data usable
on the Mac WITHOUT committing raw `.db` files.

## 1. Windows-only inventory

| File / location | What's Windows-only | Port strategy |
|---|---|---|
| `common/window.py` | `pygetwindow.getAllWindows()` — Windows-only in practice (macOS backend mostly `NotImplementedError`). Only API the bots use: `get_bounds(title_substring) -> (left, top, width, height)`. | Platform-dispatched backend: keep the Windows path, add a Quartz path on `darwin` (section 2.1). Guard the `pygetwindow` import by platform — the top-level import is the one thing that hard-breaks macOS today. |
| `common/capture.py` | Not Windows-only per se, but `mss.grab()` on Retina returns 2x physical pixels for point-sized regions — every downstream consumer assumes capture px == window px. | Normalize at capture: resize grabs to the requested logical size (section 2.2). |
| `common/input.py` | `pyautogui.FAILSAFE` + corner points from mss monitor geometry. | Portable: macOS `CGDisplayBounds` (what mss uses) is in points, same space as `pyautogui.position()`. Verify corner alignment on the Mac; no code change expected. |
| `common/score_ocr.py` | tesseract fallback paths are `C:\Program Files\...` under win32; `shutil.which` runs first everywhere. | `brew install tesseract` puts it on PATH. Optionally add darwin fallbacks (`/opt/homebrew/bin`, `/usr/local/bin`). Low priority. |
| `common/observer.py` | `ctypes.windll` click-through overlay — already `win32`-guarded. | Runs without click-through on macOS. Tooling, not bot-critical. |
| `common/idleon_save.py` | `%APPDATA%\legends-of-idleon\...\leveldb` + plyvel (dep marker already win32-only). | Launcher item, defer. macOS path is `~/Library/Application Support/legends-of-idleon/Local Storage/leveldb` (verify). Degrades gracefully today. |
| `ui/launcher.py` | `taskkill /F /T` (has `terminate()` else-branch), `CREATE_NO_WINDOW` (guarded). | Defer; if orphaned bots appear use `start_new_session=True` + `os.killpg` on POSIX. |
| `pyproject.toml` | `plyvel-ci; sys_platform == 'win32'` correctly marked. | Add `pyobjc-framework-Quartz; sys_platform == 'darwin'` for the window backend; `uv lock` resolves universally from Windows. |
| Templates (hoops/darts/chopping assets) | Captured at 1x through the Windows pipeline. | Covered by capture normalization; recapture individual templates only if confidence is poor on the Mac. |
| Path handling | Clean — pathlib relative paths throughout. | Nothing to do. |
| `README.md` platform table | macOS ❌ Untested. | Flip once one bot runs end-to-end. |

## 2. Design decisions

### 2.1 Window lookup: Quartz backend in `common/window.py`

Same contract, dispatched on `sys.platform == "darwin"`:
`Quartz.CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly |
kCGWindowListExcludeDesktopElements, kCGNullWindowID)`, match the title
substring case-insensitively against `kCGWindowName` with a
`kCGWindowOwnerName` fallback (window names are hidden until Screen
Recording permission is granted; the owner/process name — likely
"LegendsOfIdleon" — works regardless). `kCGWindowBounds` is in points
with a top-left origin in global coords — the same space pyautogui and
mss region requests use on macOS, so the window-relative coordinate
convention carries over unchanged. Move `import pygetwindow` inside the
Windows branch.

### 2.2 Retina scaling: normalize at capture (THE decision)

The repo-wide invariant: 1 captured pixel == 1 window coordinate unit
== 1 pyautogui click unit. Retina breaks it (mss returns 2x physical
pixels for a point-sized region), which would corrupt four things at
once: detected→click coords, every magic-number threshold (hoop_y ≥
530, vy bands, chopping pixel counts), DB unit comparability across
machines, and template matching (1x templates vs 2x frames falls
outside DEFAULT_SCALES' 1.5 ceiling).

**Decision: rescale grabs to the requested logical size inside
`common/capture.py`** (`cv2.resize(..., INTER_AREA)` when
`frame.shape[1] != width`; same in `grab_fullscreen`). One chokepoint,
zero downstream changes, Windows- and Mac-collected DB rows stay in the
same units, and matchTemplate cost doesn't quadruple (the poll loops
are compute-bound). Derive the scale per-grab (`frame.shape[1]/width`),
don't hardcode 2.0 (external 1x monitors, fractional scaling).
Rejected alternative (capture native 2x + per-resolution templates +
divide coords before clicking): touches every detector, threshold,
and the DB unit convention.

### 2.3 Failsafe parity

`pyautogui.FAILSAFE` + corner points should work as-is (mss monitor
dicts are points on macOS). Verify on the Mac: slam cursor to a corner
during a `check_failsafe()` loop → must abort.

### 2.4 Permissions (one-time, at the Mac, granted to the terminal app)

1. **Screen Recording** — required by mss (without it: wallpaper-only
   grabs, hidden window titles).
2. **Accessibility** — required by pyautogui (without it clicks
   silently no-op).
3. **Input Monitoring** — pynput listeners only (observe/trace tooling
   + mining), not needed for chopping/darts/hoops.

Restart the terminal fully after granting. The smoke script in the
checklist triggers all prompts deliberately.

## 3. Cross-machine data shipping (no raw .db commits)

### 3.1 What exists

`.db` files are gitignored. Hoops already ships
`minigames/hoops/assets/shots_snapshot.json` (written by
`scripts/dump_shots.py`, auto-committed at session end via
`common/auto_commit.py`). **The gap: snapshots are reviewer-facing
only** — every predictor trains directly from the local DB (hoops
`fetch_makes`/`fetch_vy_labeled`; darts `fetch_stripe_rows` at session
start), so a fresh clone starts from zero. The design move: make
snapshots a training input.

### 3.2 Darts (priority — E[stripe] is the active model)

- `scripts/dump_darts.py` → `minigames/darts/assets/throws_snapshot.json`
  (tracked). Per record: `session_started`, `throw_idx` (dedupe key),
  `wind_x`, `wind_y`, `vy_px_s` (legacy px/poll rows converted at
  export using the session's poll gaps — the snapshot ships fit-ready
  rows so the bulky polls table stays local), `pose_y`, `hit`,
  `score_increment`, `code_commit`, `aim_mode`. Header with totals +
  per-session counts (doubles as the review view).
- Split `fetch_stripe_rows` into `rows_from_db(conn)` +
  `rows_from_snapshot(path)` + a combined fetch that unions them,
  DB winning on `(session_started, throw_idx)` collisions. `main.py`
  fits from the combined fetch: no behavior change on Windows
  (snapshot ⊆ DB), Mac fits from shipped rows on day one.
- Session-end auto-commit of the snapshot, copying the hoops pattern —
  runs on both machines, so Mac data ships back the same way.
- Tests mirror `tests/test_dump_shots.py`: dump→read round-trip equals
  DB fetch; dedupe; legacy unit conversion preserved.

### 3.3 Hoops (second)

The existing snapshot's `makes` records already carry what
`fetch_makes` needs, except `clean_make` (the dump selects `made=1`
but training uses `clean_make=1` — rattle-ins must not train). Add the
flag to the dump + a combined fetch. Accepted limitation:
`fetch_clean_trajectories`/`fetch_vy_labeled` need miss rows the
makes-only snapshot lacks — snapshot-trained `gp` is sufficient for a
Mac port; extend only if the #38 model needs cross-machine data.

### 3.4 Chopping

No fitted model consumes chopping.db — nothing to ship. Skip.

### 3.5 Rejected alternatives

git-lfs .db commits (binary churn, conflicts — the stated non-goal),
`sqlite3 .dump` SQL text (AUTOINCREMENT diff noise + ships the polls
table), out-of-band sync (invisible to review). The snapshot pattern is
proven in-repo, diffable, auto-committed, and reviewable.

## 4. Ordered checklist

First bot: **chopping** — HSV masking, no templates in the main loop,
no trained model, round-end is bar-disappearance. Darts (pose template
+ digit OCR + wind geometry) and hoops (4 templates + OCR + trained
predictor) are far more sensitive to a new capture pipeline.

[code] = doable on Windows and pushed; [mac] = needs the maintainer at
the MacBook.

1. [code] Platform-dispatch `common/window.py` (guard pygetwindow
   import, add `_get_bounds_quartz`); add
   `pyobjc-framework-Quartz; sys_platform == 'darwin'`; `uv lock`.
2. [code] Retina normalization in `common/capture.py` + unit test with
   a synthetic 2x array.
3. [code] Optional darwin tesseract fallbacks in `score_ocr.py`.
4. [code] Darts data shipping (section 3.2); dump once on Windows and
   commit the first `throws_snapshot.json`.
5. [code] Hoops `clean_make` in the dump + combined fetch (3.3).
6. [mac] Clone, install uv, `uv sync`; run a smoke probe (get_bounds →
   bounds sane; grab_fullscreen → PNG looks right and comes back at
   logical size; pyautogui.position) to trigger the permission prompts
   deliberately. Note the actual `kCGWindowOwnerName` of Idleon.
7. [mac] Failsafe parity check (corner abort).
8. [mac] Chopping setup: try the existing tracked regions.json first
   (fractions transfer if window proportions match); re-pick + run
   `chopping-calibrate` if masks misalign.
9. [mac] Run `chopping` end-to-end — one full round with clicks
   landing and rows logged. **This is the "supported" gate.**
10. [code] Flip the README macOS row with caveats (permissions,
    launcher untested); document the capture normalization.
11. [mac] Darts second: existing templates first; recapture
    release.png / re-pick wind+score regions only if confidences are
    weak; verify the stripe model fits from the shipped snapshot with
    an empty local DB.
12. [mac] Hoops last (most templates + OCR).
13. Deferred launcher items: tkinter under uv CPython, POSIX
    process-tree kill, plyvel darwin marker + Application Support save
    path, overlay click-through.

Open questions to settle at the Mac (cheap probes): Idleon's
`kCGWindowOwnerName`; mss monitor dicts in points on this OS version;
whether Mac rendering shifts HSV enough to need re-tuned chopping
ranges.
