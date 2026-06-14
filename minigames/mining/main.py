"""Mining minigame bot.

Loop: detect cart + next obstacle, click to jump when an approaching pit
is within the trigger-distance window. First-pass policy is jump-only —
no slam — the goal of this version is just "don't fall into the first
pit." Tune JUMP_TRIGGER_MIN/MAX and JUMP_COOLDOWN_S empirically.

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
    read_score_from_crop, score_crop,
    _find_plank_top_y, _find_plank_x_range,
)
from minigames.mining.jump_log import open_db, log_jump, log_run, set_outcome
from minigames.mining.digit_capture import DigitCapturer

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
MINING_DB = _HERE / "assets" / "mining.db"

WINDOW_TITLE = "Legends Of Idleon"
# Detection (~35-40ms/frame after the loop-speed fix) now paces the loop,
# so keep the explicit sleep small — a big sleep would throw away the FPS
# the tracking optimization bought. A tiny yield keeps CPU sane and lets
# the human-click listener thread run.
POLL_INTERVAL = 0.005

# Click policy. Trigger when a pit is detected at distance in this range.
# Picked from the trace data: scroll speed is ~80-93 px/s leftward, so
# a 40-90 px window is roughly 0.5-1.0s of warning — enough time for the
# click + jump-physics to clear the pit.
JUMP_TRIGGER_MIN = 40
JUMP_TRIGGER_MAX = 90

# After a jump click, ignore further triggers for this long so we don't
# spam clicks while the same pit is still in the trigger window.
JUMP_COOLDOWN_S = 0.6

# How long after a jump click before we measure the outcome.
OUTCOME_DELAY_S = 1.8

# How long the bot keeps running after losing sight of the plank before
# giving up — covers brief transitions in/out of the minigame UI.
PLANK_LOST_TIMEOUT_S = 8.0

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
    args = parser.parse_args()
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        try:
            _run_inner(watch=args.watch)
        finally:
            # The DB is tracked so other machines get session data via
            # `git pull`. Rollback-journal mode keeps the on-disk file
            # consistent between transactions, so this is safe even if
            # _run_inner bailed before conn.close().
            commit_file_if_changed(
                _HERE.parent.parent,
                "minigames/mining/assets/mining.db",
                "chore(mining): refresh mining.db (auto)",
            )


def _run_inner(watch: bool = False):
    mode = "WATCH (observe-only)" if watch else "AUTO (jump policy)"
    print(f"Mining bot starting — {mode} — tracking window {WINDOW_TITLE!r}.")
    print("Move mouse to any screen corner to abort.")
    time.sleep(2)

    try:
        win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
    except WindowNotFoundError as e:
        print(e)
        return

    conn = open_db(MINING_DB)
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

    # State the human-click listener needs to read. Captured in a closure;
    # mutated in the main loop each tick. The listener thread reads the
    # latest detector state when a click happens.
    detector_state: dict = {
        "plank_y": None,
        "cart": None,
        "terrain": None,
        "plank_range": None,
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
        plank_y = _find_plank_top_y(frame)
        cart_det = find_cart_detailed(frame, plank_y=plank_y, prior=cart_track)
        cart_track = cart_det
        cart = cart_det["center"] if cart_det else None
        cart_right = (cart_det["center"][0] + cart_det["half_width"]) if cart_det else None
        now = time.time()
        plank_range = _find_plank_x_range(frame, plank_y) if plank_y else None
        terrain = (find_next_terrain(frame, cart, plank_y=plank_y, cart_right=cart_right)
                   if cart is not None else None)

        # --- FIRE FIRST: decide + click immediately after detection, before
        # any bookkeeping (state publish, settle, score read, digit capture,
        # telemetry). The sampled world state ages ~94 px/s, so a DB insert
        # or disk write between sample and click biases the jump late —
        # issue #5 / CLAUDE.md "Click timing". Click position is decorative:
        # Idleon reads a click as a button press, not a pointer (CLAUDE.md
        # "Idleon clicks are buttons"), so the cart coords are just a sane
        # in-play-area default. Bookkeeping for the fired jump happens below.
        jump_to_log = None
        if (not watch and cart is not None and plank_y is not None
                and terrain is not None and terrain["kind"] == "pit"
                and JUMP_TRIGGER_MIN <= terrain["distance_px"] <= JUMP_TRIGGER_MAX
                and now - last_click_time >= JUMP_COOLDOWN_S):
            bot_click(win_left + cart[0], win_top + cart[1])
            last_click_time = now
            jump_idx += 1
            jump_to_log = jump_idx

        # Publish detector state for the human-click listener thread to
        # consume on its next click event.
        with state_lock:
            detector_state["plank_y"] = plank_y
            detector_state["cart"] = cart
            detector_state["plank_range"] = plank_range
            detector_state["terrain"] = terrain
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
                plank_y=plank_y,
                plank_x_left=plank_range[0] if plank_range else None,
                plank_x_right=plank_range[1] if plank_range else None,
                window_w=win_w,
                window_h=win_h,
                code_commit=code_commit,
                source="bot",
            )
            print(f"JUMP #{jump_to_log} pit_dist={terrain['distance_px']} "
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
            print(f"  [t+{now - last_plank_seen:5.2f}] "
                  f"plank_y={plank_y} cart={cart} pts={last_score} "
                  f"next={terrain} (~{fps:.0f}fps)")
        last_loop_at = now

        if cart is None:
            # Run-end detection (#6): once the cart is gone, the run is
            # ending. The positive signal is the "Play Game" prompt
            # returning — faster and cleaner than waiting out the
            # plank-lost timeout. (Validated against trace_20260515: the
            # button reappears ~frame 56, vs an 8s timeout.)
            if plank_ever_seen and find_play_button(frame) is not None:
                with pending_lock:
                    for p in pending_outcomes:
                        set_outcome(conn, p["row_id"], "died",
                                    int((time.time() - p["click_time"]) * 1000))
                log_run(conn, session_started=session_started,
                        attempt_idx=attempt_idx,
                        ended_at=datetime.now().isoformat(timespec="seconds"),
                        final_score=last_score, end_reason="play_button",
                        code_commit=code_commit)
                print(f"Run ended — Play Game prompt returned (final PTS={last_score}). Exiting.")
                _print_summary(conn, session_started, attempt_idx)
                _finalize_capture(capturer)
                if listener is not None:
                    listener.stop()
                conn.close()
                return
            stale = time.time() - last_plank_seen > PLANK_LOST_TIMEOUT_S
            if watch and not plank_ever_seen:
                # Watch mode hasn't started yet — keep waiting for the user
                # to open and start the minigame; don't time out before play
                # begins (they need time to navigate + click Play themselves).
                last_plank_seen = time.time()
            elif stale:
                with pending_lock:
                    for p in pending_outcomes:
                        set_outcome(conn, p["row_id"], "died",
                                    int((time.time() - p["click_time"]) * 1000))
                if plank_ever_seen:
                    log_run(conn, session_started=session_started,
                            attempt_idx=attempt_idx,
                            ended_at=datetime.now().isoformat(timespec="seconds"),
                            final_score=last_score, end_reason="plank_lost",
                            code_commit=code_commit)
                print(f"Plank lost for >{PLANK_LOST_TIMEOUT_S}s — exiting.")
                if not plank_ever_seen and last_frame is not None:
                    _dump_diagnostics(last_frame, win_w, win_h)
                _print_summary(conn, session_started, attempt_idx)
                _finalize_capture(capturer)
                if listener is not None:
                    listener.stop()
                conn.close()
                return
            time.sleep(POLL_INTERVAL)
            continue
        last_plank_seen = time.time()
        plank_ever_seen = True
        # The jump decision + click already happened up top (fire-first),
        # before the per-frame bookkeeping.

        time.sleep(POLL_INTERVAL)


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
            plank_y=s["plank_y"],
            plank_x_left=s["plank_range"][0] if s["plank_range"] else None,
            plank_x_right=s["plank_range"][1] if s["plank_range"] else None,
            window_w=ww,
            window_h=wh,
            code_commit=code_commit,
            source="human",
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
