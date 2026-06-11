# macOS port plan + cross-machine data shipping

Status: plan (port not implemented; survey by subagent 2026-06-11).
Target: the repo cloned on a MacBook Air (Apple Silicon); goal (1) bots
runnable on macOS, README platform table flipped to supported once one
bot runs end-to-end. Goal (2) — Windows-collected training data usable
on the Mac — was resolved 2026-06-11 by committing the raw `.db` files
(see section 3); a fresh clone has the full training data.

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

## 3. Cross-machine data shipping — SUPERSEDED 2026-06-11

The snapshot-as-training-input design originally planned here (darts
`throws_snapshot.json`, hoops `clean_make` in the dump, combined
DB+snapshot fetches) is unnecessary: the raw `.db` files are now
tracked in git and auto-committed at each bot's session end (27d3857).
Every predictor trains from the local DB as before, and a `git pull`
on the Mac brings the full data — including the miss rows and polls
context a makes-only snapshot would have dropped.

This section had rejected raw `.db` commits over binary churn and
conflicts. Both concerns dissolved on inspection: the DBs total
~2.2 MB, and the data flow is one-way (bots only run on the Windows
box; the Mac is a read-only consumer), so merge conflicts can't occur.
If bots ever run on a second machine, revisit — concurrent writers
would resurrect the conflict problem, and the snapshot/union design
above is the fallback. `shots_snapshot.json` stays as a
reviewer-facing aggregate, not a training input.

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
4. ~~Darts data shipping~~ — obsolete, DBs are tracked (section 3).
5. ~~Hoops `clean_make` dump~~ — obsolete, same.
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
    weak; verify the stripe model fits from the pulled `darts.db`.
12. [mac] Hoops last (most templates + OCR).
13. Deferred launcher items: tkinter under uv CPython, POSIX
    process-tree kill, plyvel darwin marker + Application Support save
    path, overlay click-through.

Open questions to settle at the Mac (cheap probes): Idleon's
`kCGWindowOwnerName`; mss monitor dicts in points on this OS version;
whether Mac rendering shifts HSV enough to need re-tuned chopping
ranges.
