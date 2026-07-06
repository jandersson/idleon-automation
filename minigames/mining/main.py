"""Mining minigame bot.

Loop: detect cart + next obstacle, then per frame either JUMP (grounded
click — clear an approaching pit, or get airborne ahead of an ore) or SLAM
(airborne click — drop onto ore to score; the rebound is a free jump that
chains into the next slam). Trigger windows are distances tuned at a
reference scroll speed and scaled live by a ScrollVelocity estimate, since
the track accelerates within a run (see policy.py).

Every click is logged to assets/mining.db with the detector state at
fire time. After OUTCOME_DELAY_S the outcome (survived/died/unknown) is
back-filled. Query the DB to find the distance windows that actually
work: see jump_log.survival_rate_by_distance.

`mining --watch` runs an observe-only mode (no Play-Game click, no
auto-jump) that logs human clicks + captures PTS digit crops — the
data-collection vehicle for tuning the policy and bootstrapping the OCR.
See docs/mining_plan.md for the full plan (mechanic, phases, #5/#6).

Run flow:
    1. User opens the mining minigame so the "Play Game" prompt is visible
    2. uv run mining
    3. Bot clicks Play Game (requires start_button region — pick via
       mining-pick-start-button beforehand)
    4. Bot enters the detection loop. When find_next_terrain reports a pit
       with distance in [JUMP_TRIGGER_MIN, JUMP_TRIGGER_MAX], it clicks
       once at the cart's position to jump.
    5. Loop continues until the minigame ends (plank goes off-screen) or
       the user slam-corners the mouse.
"""
import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import pynput

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import check_failsafe, click as bot_click
from common.regions import get_region
from common.session_log import session_log
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from common.window import get_bounds, WindowNotFoundError
from minigames.mining.detector import (
    find_cart_detailed, find_next_terrain, find_play_button,
    read_score_from_crop, score_crop, CART_MAX_X_JUMP_PX,
    _find_plank_top_y, _find_plank_x_range,
)
from minigames.mining.jump_log import open_db, log_jump, log_run, set_outcome
from minigames.mining.digit_capture import DigitCapturer
from minigames.mining.policy import (
    GroundedBaseline, ScrollVelocity, should_jump, should_slam,
    is_cart_airborne, scale_window, scale_slam_dist,
    SCROLL_V_REF_PIT, SCROLL_V_REF_ORE,
)

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
MINING_DB = _HERE / "assets" / "mining.db"

WINDOW_TITLE = "Legends Of Idleon"
# Detection (~35-40ms/frame after the loop-speed fix) now paces the loop,
# so keep the explicit sleep small — a big sleep would throw away the FPS
# the tracking optimization bought. A tiny yield keeps CPU sane and lets
# the human-click listener thread run.
POLL_INTERVAL = 0.005

# Click policy. Trigger when a pit's leading edge is within this distance
# (px) of the cart. The bot fires on the first frame the pit enters the
# window, i.e. at ~JUMP_TRIGGER_MAX.
#
# ALL trigger constants below are distances AT THEIR REFERENCE SPEED
# (policy.SCROLL_V_REF_PIT / _ORE) and are scaled by the live
# ScrollVelocity estimate each frame — the track accelerates within a run,
# so a fixed px window drifts off the time bounds it encodes (#53).
#
# Retuned 2026-06-16 (#5) from the first-pit death trace
# (session_20260616_131320), worked through in ABSOLUTE time. Once the cart
# launches its airborne window is fixed: launch ~delta=0.23s after the
# click, airborne A~0.88s. The gap reaches the cart at a fixed world-time no
# matter when we jump; firing only shifts the window. To land on solid
# ground the window [launch, land] must CONTAIN the gap passage [near, far].
# With scroll v~79 px/s that pins two bounds on the fire distance D:
#   (1) airborne before the gap arrives:  D >= delta*v       ~= 18
#   (2) gap clears before landing:        D + W <= v*(delta+A) ~= 88   (W = gap width)
# The old [30,50] fired at ~49: D+W = 49+52 = 101 > 88, so the cart landed
# ~13px into the far edge and died (confirmed in the trace). The fix is to
# fire LATER (smaller D) — NOT earlier as an earlier note speculated; firing
# earlier worsens bound (2). [20,30] fires at ~30, inside the point-model
# feasible band [18, 88-W].
#
# CAVEAT — this is an EXPERIMENT, not a settled fix. Feasibility depends on
# the (still unmeasured) gap width W and the collision model. Center/point
# model with W~=52: a single jump clears at this timing. Full-footprint
# model (cart ~36px wide, any overlap kills): gap+footprint (~88px) exceeds
# the airborne relative travel (v*A ~= 70px) by ~19px, so NO single-jump
# timing clears and the real fix is a multi-jump (land-and-re-jump) policy.
# This window is the discriminator: a --save-frames re-run that CLEARS
# confirms the point model; one that still dies — now landing at the far
# lip, not mid-gap — confirms footprint + W~=52. Measure the true W on that
# run (the [traj] log lines below print the live arc).
JUMP_TRIGGER_MIN = 20
JUMP_TRIGGER_MAX = 30

# After a jump click, ignore further triggers for this long so we don't
# spam clicks while the same pit is still in the trigger window.
JUMP_COOLDOWN_S = 0.6

# Ore jump-slam (#5 / Phase D) — see policy.should_jump/should_slam and the
# detector ORE_* block. Tuned from the scoring run botrun_20260616_135951:
# the human jumped at ore_dist~69 and slammed ~0.75s later at ore_dist~10.
# Jump for ore in a FARTHER window than pits (so the cart is airborne by the
# time the ore scrolls under it), then SLAM when the ore reaches the cart's
# leading edge. EXPERIMENTAL — detection validated offline on that run; the
# slam timing is a first cut (n=2) the next run's kind='ore'/action='slam'
# rows will refine.
ORE_JUMP_TRIGGER_MIN = 40
ORE_JUMP_TRIGGER_MAX = 70
# Slam when the airborne cart is nearly OVER the ore. The first auto run
# (botrun_20260616_152324) slammed at ore_dist=17 and died — the cart came
# down in front of the ore and the ore hit its side. The scoring run's
# SUCCESSFUL human slams were at ore_dist <=10 (the crash was at 13), so the
# slam must fire later, when the ore is at the cart's leading edge. find_next_
# terrain reports ore no nearer than SCAN_BUFFER_PX(=10), so 11 fires it in
# the ~10-11 success zone. (If it now misses the slam — never fires, runs in
# grounded — the window is too tight and SCAN_BUFFER must drop so the ore is
# seen closer.)
ORE_SLAM_MAX_DIST = 11     # airborne + nearest ore within this px => slam now
SLAM_COOLDOWN_S = 0.5      # one slam per airborne arc

# How long after a jump click before we measure the outcome.
OUTCOME_DELAY_S = 1.8

# How long the bot keeps running after losing sight of the plank before
# giving up — covers brief transitions in/out of the minigame UI.
PLANK_LOST_TIMEOUT_S = 8.0

# Consecutive cart-miss frames before the column-search prior is dropped to
# allow a fresh full-frame re-acquisition. Below this, a transient miss keeps
# the prior anchored at the cart's fixed x (so a momentary undetected pose
# can't trigger a full search that adopts an off-plank false match). ~20
# frames is ~0.8s at the live FPS — by then the cart is genuinely gone (run
# ending) or a bad initial lock should self-heal.
CART_REACQUIRE_MISSES = 20

# Print a one-line telemetry every TELEMETRY_INTERVAL_S so we can see
# what the detector is reporting even when no jump fires.
TELEMETRY_INTERVAL_S = 0.5

# Where to dump the last-seen frame + a debug overlay if the bot exits
# without ever finding the plank — helps diagnose detection failures
# in new environments.
DIAG_DIR = _HERE / "assets" / "diagnostics"


def run():
    parser = argparse.ArgumentParser(description="Mining minigame bot")
    parser.add_argument(
        "--watch", action="store_true",
        help="Observation mode: the bot does NOT click Play Game or "
             "auto-jump. It runs the detector, logs every human click with "
             "the detector state at fire time, and captures PTS digit crops. "
             "Use for data collection so the untuned bot doesn't burn a "
             "scarce daily attempt.",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Also write every frame to assets/captures/botrun_<stamp>/ "
             "(gitignored) for offline analysis — extracting a jump-pose "
             "cart template, render_overlay QA, diagnosing what killed a "
             "run. Adds a little disk I/O; placed after the click so it "
             "doesn't delay the fire.",
    )
    args = parser.parse_args()
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        # Own the DB connection here so it's closed in a finally even if
        # _run_inner raises (failsafe abort, detector error) — the handle is
        # released and writes flushed before the auto-commit touches the file.
        conn = open_db(MINING_DB)
        try:
            _run_inner(conn, watch=args.watch, save_frames=args.save_frames)
        finally:
            conn.close()
            # The DB is tracked so other machines get session data via
            # `git pull`. Rollback-journal mode keeps the on-disk file
            # consistent between transactions, so this is safe even on abort.
            commit_file_if_changed(
                _HERE.parent.parent,
                "minigames/mining/assets/mining.db",
                "chore(mining): refresh mining.db (auto)",
            )


def _run_inner(conn, watch: bool = False, save_frames: bool = False):
    mode = "WATCH (observe-only)" if watch else "AUTO (jump policy)"
    print(f"Mining bot starting — {mode} — tracking window {WINDOW_TITLE!r}.")
    print("Move mouse to any screen corner to abort.")
    time.sleep(2)

    try:
        win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
    except WindowNotFoundError as e:
        print(e)
        return

    session_started = datetime.now().isoformat(timespec="seconds")
    code_commit = current_code_commit(_HERE.parent.parent)
    attempt_idx = 1
    jump_idx = 0
    pending_outcomes: list[dict] = []
    print(f"Mining DB: {MINING_DB} (session={session_started})")

    # Per-run digit-template capture (#6): save each distinct PTS crop so
    # digits 1-9 can be bootstrapped offline. See docs/mining_plan.md.
    capture_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capturer = DigitCapturer(_HERE / "assets" / "digit_capture" / capture_stamp)

    # Optional full-frame capture (--save-frames) for offline analysis:
    # jump-pose template extraction, render_overlay QA, death diagnosis.
    frames_dir = None
    if save_frames:
        frames_dir = _HERE / "assets" / "captures" / f"botrun_{capture_stamp}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to {frames_dir} (--save-frames)")

    if watch:
        print("Watch mode: the bot will NOT click Play Game or auto-jump. "
              "Open the minigame and play yourself — every click is logged "
              "with the detector state at fire time, and PTS digit crops are "
              "captured for the OCR bootstrap.")
    else:
        # Click Play Game button to start the attempt.
        if not _click_start_button(win_left, win_top, win_w, win_h):
            return

    start_time = time.time()
    frame_i = 0
    last_click_time = 0.0
    last_slam_time = 0.0
    last_plank_seen = time.time()
    last_telemetry_at = 0.0
    last_loop_at = 0.0  # for live FPS in telemetry
    last_frame: cv2.Mat | None = None  # for diagnostic dump on exit
    plank_ever_seen = False
    last_score: int | None = None  # most recent in-play PTS read (#6)
    # Last cart detection, fed back as `prior` so find_cart_detailed tracks
    # the cart in a narrow column instead of full-frame searching (issue #1:
    # cart x is fixed). None when the cart is lost → next detection is a
    # clean full search.
    cart_track: dict | None = None
    cart_misses = 0  # consecutive frames with no cart detection (re-acquire gate)
    # Rolling estimate of the cart's grounded (resting) y, so the fire
    # predicate can suppress a click while the cart is airborne — an airborne
    # click is a SLAM, which drives the cart down into an approaching pit
    # (the spaced-obstacle death). See minigames/mining/policy.py.
    grounded = GroundedBaseline()
    # Live scroll-speed estimate (px/s). The trigger constants above are
    # distances tuned at a reference speed, but the track ACCELERATES within
    # a run (~82 -> ~117 px/s over the scoring run) — the windows are scaled
    # by v/v_ref each frame so their time behaviour holds through the ramp.
    scroll = ScrollVelocity()

    # State the human-click listener needs to read. Captured in a closure;
    # mutated in the main loop each tick. The listener thread reads the
    # latest detector state when a click happens.
    detector_state: dict = {
        "plank_y": None,
        "cart": None,
        "cart_pose": None,
        "terrain": None,
        "plank_range": None,
        "scroll_v": None,
        "score": None,
        "win_left": 0, "win_top": 0, "win_w": 0, "win_h": 0,
    }
    state_lock = threading.Lock()
    pending_lock = threading.Lock()  # pending_outcomes touched from listener too
    # The human-click listener only runs in watch mode. In auto mode the
    # bot's own pyautogui clicks trip the listener too, double-logging each
    # bot jump as a phantom source='human' row (seen in the 2026-06-15 run).
    # Auto runs log their jumps directly; human interventions aren't a case
    # we need there.
    listener = None
    if watch:
        listener = _start_human_click_listener(
            conn, session_started, attempt_idx,
            detector_state, state_lock, pending_outcomes, pending_lock,
            code_commit,
        )

    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError:
            time.sleep(0.5)
            continue

        frame_bgra = grab_region(win_left, win_top, win_w, win_h)
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        last_frame = frame
        frame_i += 1

        # --- Detection: ONE plank/cart/terrain pass per frame, reused
        # everywhere below. find_cart_detailed tracks the cart in a narrow
        # column via the prior, and passing plank_y + cart_right into
        # find_next_terrain skips the redundant full cart search it used to
        # do. This took the loop from ~2 FPS to ~25-30 FPS (validated on
        # trace_20260515) — the resolution #5's click policy needs.
        # Anchor the plank search to the cart's grounded y (from prior frames'
        # baseline) so a competing tan band — e.g. an overworld floor when the
        # minigame is opened in a different world room — can't outscore the
        # real plank in the global window (it picked y=297 over the true 190
        # on botrun_20260615_031952, breaking terrain + score). None until the
        # baseline warms up, falling back to the global brightest-band scan.
        plank_y = _find_plank_top_y(frame, near_y=grounded.baseline())
        cart_det = find_cart_detailed(frame, plank_y=plank_y, prior=cart_track,
                                      max_x_jump=CART_MAX_X_JUMP_PX)
        # Anchor the column-search prior at the cart's fixed x: keep the last
        # good detection as the prior on a transient miss instead of resetting
        # to None (which would full-frame search and risk adopting an
        # off-plank false match — the x=631 cave/chain lock seen
        # 2026-06-15). Re-acquire from scratch only after a long miss run
        # (cart genuinely gone / a bad initial lock self-heals).
        if cart_det is not None:
            cart_track = cart_det
            cart_misses = 0
        else:
            cart_misses += 1
            if cart_misses >= CART_REACQUIRE_MISSES:
                cart_track = None
        cart = cart_det["center"] if cart_det else None
        cart_right = (cart_det["center"][0] + cart_det["half_width"]) if cart_det else None
        now = time.time()
        plank_range = _find_plank_x_range(frame, plank_y) if plank_y else None
        terrain = (find_next_terrain(frame, cart, plank_y=plank_y,
                                     cart_right=cart_right, plank_range=plank_range)
                   if cart is not None else None)
        # Track the cart's resting y for the airborne guard (updated before
        # the fire decision so it reflects this frame).
        grounded.update(cart[1] if cart is not None else None)
        grounded_baseline_y = grounded.baseline()
        # Live scroll speed -> scale the trigger windows for this frame.
        # Pure arithmetic (no IO), so it doesn't violate fire-first; it has
        # to precede the decision because the decision consumes the windows.
        scroll.update(terrain, plank_y, now)
        scroll_v = scroll.velocity()
        trig_min, trig_max = scale_window(
            JUMP_TRIGGER_MIN, JUMP_TRIGGER_MAX, scroll_v, SCROLL_V_REF_PIT)
        ore_trig_min, ore_trig_max = scale_window(
            ORE_JUMP_TRIGGER_MIN, ORE_JUMP_TRIGGER_MAX, scroll_v, SCROLL_V_REF_ORE)
        slam_max_dist = scale_slam_dist(
            ORE_SLAM_MAX_DIST, scroll_v, SCROLL_V_REF_ORE)

        # --- FIRE FIRST: decide + click immediately after detection, before
        # any bookkeeping (state publish, settle, score read, digit capture,
        # telemetry). The sampled world state ages ~94 px/s, so a DB insert
        # or disk write between sample and click biases the jump late —
        # issue #5 / CLAUDE.md "Click timing". Click position is decorative:
        # Idleon reads a click as a button press, not a pointer (CLAUDE.md
        # "Idleon clicks are buttons"), so the cart coords are just a sane
        # in-play-area default. Bookkeeping for the fired jump happens below.
        jump_to_log = None
        action = None
        if not watch and cart is not None:
            if should_jump(
                    cart=cart, plank_y=plank_y, terrain=terrain, now=now,
                    last_click_time=last_click_time,
                    grounded_baseline_y=grounded_baseline_y,
                    cooldown_s=JUMP_COOLDOWN_S,
                    trig_min=trig_min, trig_max=trig_max,
                    ore_trig_min=ore_trig_min,
                    ore_trig_max=ore_trig_max):
                bot_click(win_left + cart[0], win_top + cart[1])
                last_click_time = now
                jump_idx += 1
                jump_to_log = jump_idx
                action = "jump"
            elif should_slam(
                    cart=cart, terrain=terrain, now=now,
                    last_slam_time=last_slam_time,
                    grounded_baseline_y=grounded_baseline_y,
                    slam_cooldown_s=SLAM_COOLDOWN_S,
                    slam_max_dist=slam_max_dist):
                # Second click mid-arc = slam down onto the ore (score). Fired
                # immediately like the jump; the airborne guard in should_jump
                # ensures this branch only runs while airborne.
                bot_click(win_left + cart[0], win_top + cart[1])
                last_slam_time = now
                jump_idx += 1
                jump_to_log = jump_idx
                action = "slam"

        # Save the raw frame (post-click so it never delays the fire) when
        # --save-frames is on. Named frame_NNNN.png so render_overlay can
        # consume the dir like a trace.
        if frames_dir is not None:
            cv2.imwrite(str(frames_dir / f"frame_{frame_i:04d}.png"), frame)

        # Publish detector state for the human-click listener thread to
        # consume on its next click event.
        with state_lock:
            detector_state["plank_y"] = plank_y
            detector_state["cart"] = cart
            detector_state["cart_pose"] = cart_det["template"] if cart_det else None
            detector_state["plank_range"] = plank_range
            detector_state["terrain"] = terrain
            detector_state["scroll_v"] = scroll_v
            detector_state["score"] = last_score
            detector_state["win_left"] = win_left
            detector_state["win_top"] = win_top
            detector_state["win_w"] = win_w
            detector_state["win_h"] = win_h

        # Settle any pending-outcome jumps whose measurement window expired.
        with pending_lock:
            pending_outcomes = _settle_outcomes(conn, pending_outcomes, now, cart)

        # Log the fired bot jump now (after the click — bookkeeping never
        # delays the fire). plank_range was computed pre-click from the same
        # in-memory frame, so it's identical to a post-click recompute.
        if jump_to_log is not None:
            row_id = log_jump(
                conn,
                session_started=session_started,
                attempt_idx=attempt_idx,
                jump_idx=jump_to_log,
                clicked_at=datetime.now().isoformat(timespec="milliseconds"),
                cart_x=cart[0],
                cart_y=cart[1],
                next_kind=terrain["kind"],
                next_x=terrain["x"],
                next_distance_px=terrain["distance_px"],
                next_width_px=terrain.get("width"),
                scroll_v_px_s=scroll_v,
                plank_y=plank_y,
                plank_x_left=plank_range[0] if plank_range else None,
                plank_x_right=plank_range[1] if plank_range else None,
                window_w=win_w,
                window_h=win_h,
                code_commit=code_commit,
                source="bot",
                action=action,
                cart_height_above_plank=(plank_y - cart[1]),
                cart_pose=cart_det["template"] if cart_det else None,
                score_at_click=last_score,
            )
            print(f"{action.upper()} #{jump_to_log} "
                  f"kind={terrain['kind'] if terrain else None} "
                  f"dist={terrain['distance_px'] if terrain else None} "
                  f"cart=({cart[0]},{cart[1]}) row={row_id}")
            with pending_lock:
                pending_outcomes.append({"row_id": row_id, "click_time": now})

        # Read the live PTS score while the plank is up (the readout sits
        # just below it), and feed the same crop to the digit-template
        # capturer (#6). Tracked so the final value can be logged at
        # run-end — the readout vanishes once the Play Game prompt returns.
        if plank_y is not None:
            crop = score_crop(frame, plank_y, plank_range)
            if crop is not None:
                if capturer.maybe_save(crop, frame_idx=frame_i,
                                       t_rel=now - start_time):
                    print(f"  [digit-capture] new PTS crop #{capturer.count} saved")
                score = read_score_from_crop(crop)
                if score is not None:
                    last_score = score

        # Periodic telemetry — see what the detector reports (incl. live FPS,
        # so the loop-speed fix is verifiable at the game).
        if now - last_telemetry_at >= TELEMETRY_INTERVAL_S:
            last_telemetry_at = now
            fps = (1.0 / (now - last_loop_at)) if last_loop_at else 0.0
            v_str = f"{scroll_v:.0f}" if scroll_v is not None else "warmup"
            print(f"  [t+{now - last_plank_seen:5.2f}] "
                  f"plank_y={plank_y} cart={cart} pts={last_score} "
                  f"next={terrain} v={v_str} (~{fps:.0f}fps)")
        last_loop_at = now

        # While a fired jump's outcome is pending, log the cart's arc every
        # frame (height above the plank + pit distance) so the jump-arc and
        # clearance timing can be measured at the LIVE resolution — that's
        # what pins down JUMP_TRIGGER_MIN/MAX (#5). Only active in auto mode
        # in the ~1.8s after a bot jump.
        if not watch and pending_outcomes:
            h = (plank_y - cart[1]) if (plank_y is not None and cart is not None) else None
            pd = terrain["distance_px"] if (terrain and terrain["kind"] == "pit") else None
            air = is_cart_airborne(cart[1] if cart else None, grounded_baseline_y)
            print(f"  [traj] dt=+{now - last_click_time:4.2f} height={h} "
                  f"pit_dist={pd} airborne={air} base={grounded_baseline_y}")

        if cart is None:
            # Run-end detection (#6): once the cart is gone, the run is
            # ending. The positive signal is the "Play Game" prompt
            # returning — faster and cleaner than waiting out the
            # plank-lost timeout. (Validated against trace_20260515: the
            # button reappears ~frame 56, vs an 8s timeout.)
            if plank_ever_seen and find_play_button(frame) is not None:
                print(f"Run ended — Play Game prompt returned "
                      f"(final PTS={last_score}). Exiting.")
                _end_run(conn, pending_outcomes, pending_lock, capturer,
                         listener, session_started, attempt_idx,
                         final_score=last_score, end_reason="play_button",
                         code_commit=code_commit, log_the_run=True)
                return
            stale = time.time() - last_plank_seen > PLANK_LOST_TIMEOUT_S
            if watch and not plank_ever_seen:
                # Watch mode hasn't started yet — keep waiting for the user
                # to open and start the minigame; don't time out before play
                # begins (they need time to navigate + click Play themselves).
                last_plank_seen = time.time()
            elif stale:
                print(f"Plank lost for >{PLANK_LOST_TIMEOUT_S}s — exiting.")
                if not plank_ever_seen and last_frame is not None:
                    _dump_diagnostics(last_frame, win_w, win_h)
                _end_run(conn, pending_outcomes, pending_lock, capturer,
                         listener, session_started, attempt_idx,
                         final_score=last_score, end_reason="plank_lost",
                         code_commit=code_commit, log_the_run=plank_ever_seen)
                return
            time.sleep(POLL_INTERVAL)
            continue
        last_plank_seen = time.time()
        plank_ever_seen = True
        # The jump decision + click already happened up top (fire-first),
        # before the per-frame bookkeeping.

        time.sleep(POLL_INTERVAL)


def _end_run(conn, pending_outcomes, pending_lock, capturer, listener,
             session_started, attempt_idx, *, final_score, end_reason,
             code_commit, log_the_run: bool) -> None:
    """Shared run-end teardown: settle pending jump outcomes, write the
    per-attempt runs row (skipped when play never started — log_the_run
    False), print the session summary, flush the digit-capture manifest,
    and stop the watch-mode click listener."""
    with pending_lock:
        _settle_pending_on_death(conn, pending_outcomes, time.time())
    if log_the_run:
        log_run(conn, session_started=session_started,
                attempt_idx=attempt_idx,
                ended_at=datetime.now().isoformat(timespec="seconds"),
                final_score=final_score, end_reason=end_reason,
                code_commit=code_commit)
    _print_summary(conn, session_started, attempt_idx)
    _finalize_capture(capturer)
    if listener is not None:
        listener.stop()


def _settle_outcomes(conn, pending, now, cart):
    """Walk the pending-outcome list; for any past OUTCOME_DELAY_S since
    its click, write its outcome and drop it from the list. Return the
    new pending list.

    Outcome is just "is the cart still there a beat later?": detected →
    survived (it cleared the obstacle), gone → died (it fell in). Once the
    cart is lost the attempt is over — it doesn't reappear mid-run. The old
    plank_y branch produced "unknown" because a death often leaves the
    plank briefly visible (seen 2026-06-15: the fatal jump logged
    'unknown'), which is useless for survival_rate_by_distance."""
    still_pending = []
    for p in pending:
        elapsed = now - p["click_time"]
        if elapsed < OUTCOME_DELAY_S:
            still_pending.append(p)
            continue
        outcome = "survived" if cart is not None else "died"
        set_outcome(conn, p["row_id"], outcome, int(elapsed * 1000))
        print(f"  OUTCOME row={p['row_id']}: {outcome} ({int(elapsed*1000)}ms)")
    return still_pending


def _settle_pending_on_death(conn, pending, now) -> None:
    """Force-settle every still-pending jump when the run ends. The most
    recent jump (last appended) is the fatal one; any earlier still-pending
    jumps survived — the cart was alive to make the later jump. Blanket-'died'
    would attribute one death to every distance bin those earlier jumps fired
    at, systematically depressing the survival rate of good trigger distances
    once a multi-jump policy overlaps several jumps in the settle window. With
    today's single-jump rhythm `pending` is usually one entry, so this reduces
    to 'died' — but it's correct ahead of the multi-jump policy."""
    for i, p in enumerate(pending):
        outcome = "died" if i == len(pending) - 1 else "survived"
        set_outcome(conn, p["row_id"], outcome, int((now - p["click_time"]) * 1000))


def _finalize_capture(capturer) -> None:
    """Write the digit-capture manifest and report where the crops landed.
    Crops are written to disk as they're seen (in maybe_save), so even a
    failsafe abort keeps them — this just records the manifest + summary."""
    path = capturer.flush()
    if path is not None:
        print(f"Digit capture: {capturer.count} distinct PTS crops -> "
              f"{path.parent} (label + bootstrap digits 1-9 offline; "
              f"see docs/mining_plan.md)")
    else:
        print("Digit capture: no readable PTS crops this run.")


def _print_summary(conn, session_started, attempt_idx):
    cur = conn.execute(
        '''
        SELECT outcome, COUNT(*) FROM jumps
        WHERE session_started = ? AND attempt_idx = ?
        GROUP BY outcome
        ''',
        (session_started, attempt_idx),
    )
    rows = cur.fetchall()
    if rows:
        summary = ", ".join(f"{o or 'pending'}={n}" for o, n in rows)
        print(f"Session summary (attempt {attempt_idx}): {summary}")


def _start_human_click_listener(conn, session_started, attempt_idx,
                                detector_state, state_lock,
                                pending_outcomes, pending_lock,
                                code_commit):
    """Start a pynput listener that logs every left-click inside the game
    window to mining.db with source='human'. Returns the listener; caller
    must call .stop() on exit.

    Each human click reads the most recent detector state (captured in
    the main loop) so we get cart position + nearest terrain + plank
    context at click time. Outcome is back-filled the same way bot
    clicks are."""
    jump_counter = [0]  # mutable closure cell

    def on_click(screen_x: int, screen_y: int, button, pressed: bool) -> None:
        if not pressed or button != pynput.mouse.Button.left:
            return
        with state_lock:
            s = dict(detector_state)
        wl, wt, ww, wh = s["win_left"], s["win_top"], s["win_w"], s["win_h"]
        if ww == 0 or not (wl <= screen_x <= wl + ww and wt <= screen_y <= wt + wh):
            return
        jump_counter[0] += 1
        row_id = log_jump(
            conn,
            session_started=session_started,
            attempt_idx=attempt_idx,
            jump_idx=jump_counter[0],
            clicked_at=datetime.now().isoformat(timespec="milliseconds"),
            cart_x=s["cart"][0] if s["cart"] else None,
            cart_y=s["cart"][1] if s["cart"] else None,
            next_kind=s["terrain"]["kind"] if s["terrain"] else None,
            next_x=s["terrain"]["x"] if s["terrain"] else None,
            next_distance_px=s["terrain"]["distance_px"] if s["terrain"] else None,
            next_width_px=s["terrain"].get("width") if s["terrain"] else None,
            scroll_v_px_s=s["scroll_v"],
            plank_y=s["plank_y"],
            plank_x_left=s["plank_range"][0] if s["plank_range"] else None,
            plank_x_right=s["plank_range"][1] if s["plank_range"] else None,
            window_w=ww,
            window_h=wh,
            code_commit=code_commit,
            source="human",
            action="jump",
            cart_height_above_plank=(
                s["plank_y"] - s["cart"][1]
                if (s["plank_y"] is not None and s["cart"]) else None),
            cart_pose=s["cart_pose"],
            score_at_click=s["score"],
        )
        print(f"  HUMAN click logged: row={row_id} "
              f"at ({screen_x - wl},{screen_y - wt}) "
              f"next={s['terrain']}")
        with pending_lock:
            pending_outcomes.append({"row_id": row_id, "click_time": time.time()})

    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()
    return listener


def _dump_diagnostics(frame, win_w: int, win_h: int) -> None:
    """When the bot exits without ever seeing the plank, dump the last
    frame + an annotated diagnostic image showing where the detector
    looked. Lets the user pull the saved PNG offline and see why
    detection failed."""
    from minigames.mining.detector import (
        PLANK_X0_FRAC, PLANK_X1_FRAC,
        PLANK_Y_FRAC_MIN, PLANK_Y_FRAC_MAX,
    )
    from datetime import datetime
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = DIAG_DIR / f"no_plank_{stamp}_raw.png"
    annot_path = DIAG_DIR / f"no_plank_{stamp}_annotated.png"
    cv2.imwrite(str(raw_path), frame)
    annot = frame.copy()
    y0 = int(PLANK_Y_FRAC_MIN * win_h)
    y1 = int(PLANK_Y_FRAC_MAX * win_h)
    x0 = int(PLANK_X0_FRAC * win_w)
    x1 = int(PLANK_X1_FRAC * win_w)
    cv2.rectangle(annot, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(annot, f"plank search region ({win_w}x{win_h})",
                (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
    cv2.imwrite(str(annot_path), annot)
    print(f"Diagnostic: {raw_path.name} + {annot_path.name} "
          f"(check minigames/mining/assets/diagnostics/)")


def _click_start_button(win_left, win_top, win_w, win_h) -> bool:
    """Click the Play Game button. Preferred path: template-match it
    visually so the bot survives window/environment changes. Fallback:
    a saved regions.json rectangle for legacy setups."""
    frame_bgra = grab_region(win_left, win_top, win_w, win_h)
    frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    rel = find_play_button(frame)
    if rel is not None:
        cx = win_left + rel[0]
        cy = win_top + rel[1]
        print(f"Play Game (template-matched): window-rel ({rel[0]},{rel[1]}) "
              f"-> screen ({cx},{cy})")
        bot_click(cx, cy, jitter=0)
        time.sleep(0.5)
        return True

    start_btn = get_region(_HERE, "start_button", win_w, win_h)
    if start_btn is None:
        print("Play Game not template-matched and no 'start_button' region "
              "saved. Either bring the prompt on-screen, or run "
              "mining-pick-start-button to save a fallback region.")
        return False
    cx = win_left + start_btn["left"] + start_btn["width"] // 2
    cy = win_top + start_btn["top"] + start_btn["height"] // 2
    print(f"Play Game (region fallback): rect ({start_btn['left']},"
          f"{start_btn['top']}) {start_btn['width']}x{start_btn['height']} "
          f"-> screen ({cx},{cy})")
    bot_click(cx, cy, jitter=0)
    time.sleep(0.5)
    return True


if __name__ == "__main__":
    run()
