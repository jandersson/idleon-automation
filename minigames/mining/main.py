"""Mining minigame bot.

Loop: detect cart + next obstacle, click to jump when an approaching pit
is within the trigger-distance window. First-pass policy is jump-only —
no slam — the goal of this version is just "don't fall into the first
pit." Tune JUMP_TRIGGER_MIN/MAX and JUMP_COOLDOWN_S empirically.

Every click is logged to assets/mining.db with the detector state at
fire time. After OUTCOME_DELAY_S the outcome (survived/died/unknown) is
back-filled. Query the DB to find the distance windows that actually
work: see jump_log.survival_rate_by_distance.

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
from common.window import get_bounds, WindowNotFoundError
from minigames.mining.detector import find_cart, find_next_terrain, find_play_button, _find_plank_top_y, _find_plank_x_range
from minigames.mining.jump_log import open_db, log_jump, set_outcome

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
MINING_DB = _HERE / "assets" / "mining.db"

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.03

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
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Mining bot starting — tracking window {WINDOW_TITLE!r}.")
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

    # Click Play Game button to start the attempt.
    if not _click_start_button(win_left, win_top, win_w, win_h):
        return

    last_click_time = 0.0
    last_plank_seen = time.time()
    last_telemetry_at = 0.0
    last_frame: cv2.Mat | None = None  # for diagnostic dump on exit
    plank_ever_seen = False

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

        plank_y = _find_plank_top_y(frame)
        cart = find_cart(frame)
        now = time.time()

        # Publish detector state for the human-click listener thread to
        # consume on its next click event.
        with state_lock:
            detector_state["plank_y"] = plank_y
            detector_state["cart"] = cart
            detector_state["plank_range"] = (
                _find_plank_x_range(frame, plank_y) if plank_y else None
            )
            detector_state["terrain"] = (
                find_next_terrain(frame, cart) if cart else None
            )
            detector_state["win_left"] = win_left
            detector_state["win_top"] = win_top
            detector_state["win_w"] = win_w
            detector_state["win_h"] = win_h

        # Settle any pending-outcome jumps whose measurement window expired.
        with pending_lock:
            pending_outcomes = _settle_outcomes(conn, pending_outcomes, now,
                                                cart, plank_y)

        # Periodic telemetry — see what the detector is doing when no
        # jump fires.
        if now - last_telemetry_at >= TELEMETRY_INTERVAL_S:
            last_telemetry_at = now
            terr = find_next_terrain(frame, cart) if cart else None
            print(f"  [t+{now - last_plank_seen:5.2f}] "
                  f"plank_y={plank_y} cart={cart} next={terr}")

        if cart is None:
            if time.time() - last_plank_seen > PLANK_LOST_TIMEOUT_S:
                with pending_lock:
                    for p in pending_outcomes:
                        set_outcome(conn, p["row_id"], "died",
                                    int((time.time() - p["click_time"]) * 1000))
                print(f"Plank lost for >{PLANK_LOST_TIMEOUT_S}s — exiting.")
                if not plank_ever_seen and last_frame is not None:
                    _dump_diagnostics(last_frame, win_w, win_h)
                _print_summary(conn, session_started, attempt_idx)
                listener.stop()
                conn.close()
                return
            time.sleep(POLL_INTERVAL)
            continue
        last_plank_seen = time.time()
        plank_ever_seen = True

        terrain = find_next_terrain(frame, cart)
        if (terrain is not None
                and terrain["kind"] == "pit"
                and JUMP_TRIGGER_MIN <= terrain["distance_px"] <= JUMP_TRIGGER_MAX
                and now - last_click_time >= JUMP_COOLDOWN_S):
            jump_idx += 1
            plank_range = _find_plank_x_range(frame, plank_y) if plank_y else None
            row_id = log_jump(
                conn,
                session_started=session_started,
                attempt_idx=attempt_idx,
                jump_idx=jump_idx,
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
            # Click position is decorative — Idleon treats the click as a
            # button press, not a pointer event. Aiming at the cart is
            # just a sane default that's guaranteed to be inside the play
            # area. See CLAUDE.md ("Idleon clicks are buttons, not
            # pointers"). The thing that controls jump vs slam is the
            # click *timing* relative to the cart being grounded vs
            # airborne, not the click coordinates.
            screen_x = win_left + cart[0]
            screen_y = win_top + cart[1]
            print(f"JUMP #{jump_idx} pit_dist={terrain['distance_px']} "
                  f"cart=({cart[0]},{cart[1]}) row={row_id}")
            bot_click(screen_x, screen_y)
            last_click_time = now
            with pending_lock:
                pending_outcomes.append({"row_id": row_id, "click_time": now})

        time.sleep(POLL_INTERVAL)


def _settle_outcomes(conn, pending, now, cart, plank_y):
    """Walk the pending-outcome list; for any past OUTCOME_DELAY_S since
    its click, write its outcome and drop it from the list. Return the
    new pending list."""
    still_pending = []
    for p in pending:
        elapsed = now - p["click_time"]
        if elapsed < OUTCOME_DELAY_S:
            still_pending.append(p)
            continue
        if plank_y is None:
            outcome = "died"
        elif cart is not None:
            outcome = "survived"
        else:
            outcome = "unknown"
        set_outcome(conn, p["row_id"], outcome, int(elapsed * 1000))
        print(f"  OUTCOME row={p['row_id']}: {outcome} ({int(elapsed*1000)}ms)")
    return still_pending


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
