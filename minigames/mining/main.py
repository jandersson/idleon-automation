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
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import check_failsafe, click as bot_click
from common.regions import get_region
from common.session_log import session_log
from common.shot_log import current_code_commit
from common.window import get_bounds, WindowNotFoundError
from minigames.mining.detector import find_cart, find_next_terrain, _find_plank_top_y, _find_plank_x_range
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

    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError:
            time.sleep(0.5)
            continue

        frame_bgra = grab_region(win_left, win_top, win_w, win_h)
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        plank_y = _find_plank_top_y(frame)
        cart = find_cart(frame)
        now = time.time()

        # Settle any pending-outcome jumps whose measurement window expired.
        pending_outcomes = _settle_outcomes(conn, pending_outcomes, now,
                                            cart, plank_y)

        if cart is None:
            if time.time() - last_plank_seen > PLANK_LOST_TIMEOUT_S:
                # Resolve any still-pending outcomes as died (game's gone).
                for p in pending_outcomes:
                    set_outcome(conn, p["row_id"], "died",
                                int((time.time() - p["click_time"]) * 1000))
                print(f"Plank lost for >{PLANK_LOST_TIMEOUT_S}s — exiting.")
                _print_summary(conn, session_started, attempt_idx)
                conn.close()
                return
            time.sleep(POLL_INTERVAL)
            continue
        last_plank_seen = time.time()

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
            screen_x = win_left + cart[0]
            screen_y = win_top + cart[1]
            print(f"JUMP #{jump_idx} pit_dist={terrain['distance_px']} "
                  f"cart=({cart[0]},{cart[1]}) row={row_id}")
            bot_click(screen_x, screen_y)
            last_click_time = now
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


def _click_start_button(win_left, win_top, win_w, win_h) -> bool:
    start_btn = get_region(_HERE, "start_button", win_w, win_h)
    if start_btn is None:
        print("No 'start_button' region in regions.json. "
              "Run mining-pick-start-button first.")
        return False
    cx = win_left + start_btn["left"] + start_btn["width"] // 2
    cy = win_top + start_btn["top"] + start_btn["height"] // 2
    print(f"Clicking Play Game at screen ({cx}, {cy})")
    # No jitter for the UI button click — the button sits over the
    # walkable game world, so a ±3px drift can land on bare ground and
    # move the character instead, canceling the minigame entry.
    bot_click(cx, cy, jitter=0)
    time.sleep(0.5)
    return True


if __name__ == "__main__":
    run()
