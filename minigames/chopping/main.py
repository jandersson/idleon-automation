import time
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.regions import get_region
from common.session_log import session_log
from common.git_info import current_code_commit
from common.window import get_bounds, WindowNotFoundError
from minigames.chopping.chop_log import open_db, log_chop, log_poll, set_outcome
from minigames.chopping.detector import (
    analyze_bar,
    bar_pixel_count,
    gold_distance_ahead,
    leaf_vx_px_s,
    nearest_red_distance,
    red_distance_ahead,
    zone_layout,
)

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
CHOPPING_DB = _HERE / "assets" / "chopping.db"

WINDOW_TITLE = "Legends Of Idleon"

# Regions are loaded from assets/regions.json each iteration so they survive
# window resizes. Pick via chopping-pick-bar-region / chopping-pick-button-region.

POLL_INTERVAL = 0.01

# Post-chop fire hold. A successful chop re-rolls the zone layout
# within 86-201ms (observed, 00:46 session) — that re-roll is the
# in-game ack the chop registered. But the re-roll is NOT the end of
# the game's own chop cooldown: clicks fired 198/201/225ms after a
# registered chop were silently IGNORED (no re-roll, no point), while
# 655/720/812ms gaps all registered. So the hold releases only when
# BOTH the layout has re-rolled AND MIN_INTERCHOP_S has passed; the
# 450ms value bisects the unmeasured (225, 655)ms registration
# boundary — each session's polls show whether 450ms chops register
# (re-roll follows) and the bound tightens for free.
# COOLDOWN_AFTER_CLICK is the fallback when no re-roll is ever seen
# (a 0px re-roll is possible — shifts run 1-3px).
MIN_INTERCHOP_S = 0.45
COOLDOWN_AFTER_CLICK = 0.70

# Cap on the same-sweep gold upgrade: defer a safe green fire only when
# the gold ahead is reachable within this long. Keeps a crawling leaf
# from stalling the chop rate; in practice a full bar sweep at the
# slowest observed speed (~257 px/s over 222px) is ~860ms, so this
# rarely binds mid-round.
GOLD_RIDE_MAX_MS = 1000

# If the same pointer x is reported in the same zone this many clicks in a row,
# assume the minigame is over (a stationary post-game UI element looks like a
# leaf to the detector) and exit cleanly instead of spamming clicks.
STAGNATION_LIMIT = 5

# Round-end detector. While the round runs, the bar has hundreds of colored
# pixels (green + red + occasional gold). When the round ends the bar UI
# disappears entirely and the count collapses toward zero. If we see fewer
# than BAR_DEAD_PIXEL_THRESHOLD colored pixels for BAR_DEAD_GRACE_S
# consecutive seconds, the round is over.
#
# Replaces the previous game_over.png template match (the "JUST IDLE" world-
# map button) which fired inconsistently — caught session 1 but missed
# session 2 on 2026-05-24.
BAR_DEAD_PIXEL_THRESHOLD = 30
BAR_DEAD_GRACE_S = 1.5

# Skip the click if the leaf's left edge is within this many pixels of a red
# column. Click latency (~50ms pre-click delay + OS jitter) lets the leaf drift
# a few px between detection and click — landing in red ends the minigame.
# Kept as a direction-blind floor (covers detection jitter and bounce
# reversals); the load-bearing gate is the time-to-red one below.
RED_SAFETY_MARGIN_PX = 8

# Directional safety gate: skip the click when the nearest red column
# IN THE LEAF'S DIRECTION OF TRAVEL is closer than this many ms at the
# measured leaf speed. Calibrated from the 2026-06-11 00:13 session:
# chop #3 died with red 19px ahead at ~257-386 px/s (~50-75ms), while
# chop #2 survived 8px of red BEHIND a rightward leaf — the pixel
# margin can't represent that asymmetry. 150ms = ~2x the observed kill
# latency; tighten once outcome data accumulates.
#
# The leaf ACCELERATES as the round progresses (game fact, 2026-06-11).
# The gate self-adapts — vx is measured live, so a faster leaf needs
# proportionally more red-free runway — but that also means fewer
# gate-open windows late-round as green shrinks and speed rises. If
# validation sessions show fire starvation late-round, lower this
# toward the measured kill latency rather than bypassing the gate.
MIN_TIME_TO_RED_MS = 150

# Direction-unknown fires are banned (2026-06-11, the 00:46 death):
# the leaf read the same x twice 127ms apart (stale/frozen render),
# vx came out exactly 0.0, the old `abs(vx) > 1e-6` guard silently
# disarmed the time-to-red gate, and the click landed while the real
# leaf was doing ~355 px/s with red 31px (~87ms) ahead. Below this
# speed the direction signal is jitter/turnaround/stale — wait ~one
# poll for a clean read instead of firing blind.
MIN_VX_FOR_FIRE = 30.0

# Leaf motion is EASED, not constant-speed (00:46 session: mid-bar
# ~630 px/s vs ~290-350 near the edges — sinusoidal like the hoops
# platform bob). An instantaneous vx sampled in the slow edge region
# understates the speed the leaf will reach crossing toward a mid-bar
# red, so time-to-red uses an effective speed of at least
# EASING_SPEED_FLOOR_FRAC of the fastest |vx| seen in the recent
# window (the window tracks the round's ramp, if any).
EASING_SPEED_FLOOR_FRAC = 0.5
SPEED_WINDOW_S = 3.0

# Per-poll DB write rate cap. 0 = log every loop iteration (~40-80Hz).
# Raised from 10Hz to full rate 2026-06-11: attributing the speed ramp
# (per-chop? per-bounce? eased motion?) needs leaf positions at loop
# resolution — 10Hz aliases sweeps that bounce between samples. A 60s
# round is ~4k rows / ~quarter MB, fine for SQLite; rows are committed
# in batches (per fire + at round end), not per insert.
POLL_LOG_INTERVAL = 0.0


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Chopping bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    conn = open_db(CHOPPING_DB)
    session_started = datetime.now().isoformat(timespec="seconds")
    session_start_t = time.time()
    code_commit = current_code_commit(_HERE.parent.parent)
    chop_idx = 0
    # The last logged chop sits with outcome=NULL until the next iteration
    # tells us whether the bot survived (another successful chop) or the
    # round ended (bar disappeared). pending = (row_id, click_time).
    pending: tuple[int, float] | None = None
    print(f"Chopping DB: {CHOPPING_DB} (session={session_started})")

    last_click: tuple[int, str] | None = None
    stagnation_count = 0
    last_poll_log = 0.0
    bar_dead_since: float | None = None  # wall-clock t when bar first looked dead
    # (wall_time, leaf_x) at full poll rate, for the leaf-velocity
    # estimate feeding the time-to-red gate. Cleared whenever the leaf
    # isn't detected so a respawned leaf can't inherit stale direction.
    leaf_track: list[tuple[float, int]] = []
    # Post-chop fire hold: the pre-click layout string + timestamps.
    # Fires resume when the layout has re-rolled AND MIN_INTERCHOP_S
    # has passed (or the COOLDOWN_AFTER_CLICK fallback expires).
    fire_hold_layout: str | None = None
    fire_hold_click_t = 0.0
    riding_gold = False  # print-once flag for the gold-upgrade hold
    # (wall_time, |vx|) over the recent window — the easing speed floor.
    speed_track: list[tuple[float, float]] = []

    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        bar_region = get_region(_HERE, "bar", win_w, win_h)
        leaf_region = get_region(_HERE, "leaf", win_w, win_h)
        button_region = get_region(_HERE, "button", win_w, win_h)
        if bar_region is None or button_region is None:
            print("Missing region(s) in regions.json. Run chopping-pick-bar-region and chopping-pick-button-region first.")
            time.sleep(2)
            continue

        bar_frame = grab_region(
            win_left + bar_region["left"],
            win_top + bar_region["top"],
            bar_region["width"],
            bar_region["height"],
        )
        leaf_frame = None
        if leaf_region is not None:
            leaf_frame = grab_region(
                win_left + leaf_region["left"],
                win_top + leaf_region["top"],
                leaf_region["width"],
                leaf_region["height"],
            )
        pointer_x, zone = analyze_bar(bar_frame, leaf_frame=leaf_frame)
        bar_px = bar_pixel_count(bar_frame)
        layout = zone_layout(bar_frame)  # reused by logs + the fire hold

        # Round-end check via bar disappearance — replaces game_over template.
        now = time.time()
        if bar_px < BAR_DEAD_PIXEL_THRESHOLD:
            if bar_dead_since is None:
                bar_dead_since = now
            elif now - bar_dead_since >= BAR_DEAD_GRACE_S:
                if pending is not None:
                    row_id, click_time = pending
                    set_outcome(conn, row_id, "round_ended",
                                int((now - click_time) * 1000))
                    pending = None
                print(f"Bar gone for {now - bar_dead_since:.1f}s (bar_px={bar_px}) — round ended, stopping.")
                conn.commit()
                conn.close()
                return
        else:
            bar_dead_since = None

        red_dist = None
        if pointer_x is None:
            leaf_track.clear()
        else:
            red_dist = nearest_red_distance(bar_frame, pointer_x)
            leaf_track.append((now, pointer_x))
            while leaf_track and now - leaf_track[0][0] > 0.3:
                leaf_track.pop(0)
        leaf_vx = leaf_vx_px_s(leaf_track)
        if leaf_vx is not None:
            speed_track.append((now, abs(leaf_vx)))
        while speed_track and now - speed_track[0][0] > SPEED_WINDOW_S:
            speed_track.pop(0)

        if pointer_x is not None and zone in ("green", "gold"):
            # Post-chop re-arm: layout re-rolled (chop registered) AND
            # the game's own chop-registration interval has passed —
            # clicks earlier than that are silently ignored in-game
            # (and an ignored click is pure downside: no point, still
            # evaluated against red). Fallback deadline covers a 0px
            # re-roll. The loop keeps sampling at full rate throughout.
            if fire_hold_layout is not None:
                rerolled = layout != fire_hold_layout
                interchop_ok = now - fire_hold_click_t >= MIN_INTERCHOP_S
                fallback = now - fire_hold_click_t >= COOLDOWN_AFTER_CLICK
                if (rerolled and interchop_ok) or fallback:
                    fire_hold_layout = None
                else:
                    if now - last_poll_log >= POLL_LOG_INTERVAL:
                        log_poll(conn, session_started,
                                 int((now - session_start_t) * 1000),
                                 pointer_x, zone, red_dist, bar_px, 0,
                                 zone_layout=layout, leaf_vx_px_s=leaf_vx)
                        last_poll_log = now
                    time.sleep(POLL_INTERVAL)
                    continue

            # Time-to-red gate (directional, the load-bearing one) plus
            # the legacy pixel-margin floor. Direction must be KNOWN
            # (see MIN_VX_FOR_FIRE), and the projected speed is floored
            # at a fraction of the recent max so an edge-region vx
            # sample can't understate the eased mid-bar speed.
            red_ahead = None
            time_to_red_ms = None
            direction_known = leaf_vx is not None and abs(leaf_vx) >= MIN_VX_FOR_FIRE
            if direction_known:
                red_ahead = red_distance_ahead(bar_frame, pointer_x, leaf_vx)
                if red_ahead is not None:
                    recent_max = max((v for _, v in speed_track), default=0.0)
                    eff_speed = max(abs(leaf_vx), EASING_SPEED_FLOOR_FRAC * recent_max)
                    time_to_red_ms = int(red_ahead / eff_speed * 1000)
            unsafe = (
                not direction_known
                or (red_dist is not None and red_dist < RED_SAFETY_MARGIN_PX)
                or (time_to_red_ms is not None and time_to_red_ms < MIN_TIME_TO_RED_MS)
            )
            if unsafe:
                if now - last_poll_log >= POLL_LOG_INTERVAL:
                    log_poll(conn, session_started,
                             int((now - session_start_t) * 1000),
                             pointer_x, zone, red_dist, bar_px, 0,
                             zone_layout=layout,
                             leaf_vx_px_s=leaf_vx)
                    last_poll_log = now
                time.sleep(POLL_INTERVAL)
                continue

            # Same-sweep gold upgrade: when gold lies ahead of the leaf
            # BEFORE any red, ride to it instead of taking the green —
            # +2 beats +1, gold also slows the leaf, and the same sweep
            # reaches it with no extra bounce (docs/chopping_notes.md).
            if zone == "green":  # direction is known here — unsafe gate filtered
                gold_ahead = gold_distance_ahead(bar_frame, pointer_x, leaf_vx)
                gold_ride_ms = (
                    int(gold_ahead / abs(leaf_vx) * 1000)
                    if gold_ahead is not None else None
                )
                if (
                    gold_ride_ms is not None
                    and gold_ride_ms <= GOLD_RIDE_MAX_MS
                    and (red_ahead is None or gold_ahead < red_ahead)
                ):
                    if not riding_gold:
                        riding_gold = True
                        print(f"  [aim] gold {gold_ahead}px ahead (~{gold_ride_ms}ms) "
                              f"— riding past green")
                    if now - last_poll_log >= POLL_LOG_INTERVAL:
                        log_poll(conn, session_started,
                                 int((now - session_start_t) * 1000),
                                 pointer_x, zone, red_dist, bar_px, 0,
                                 zone_layout=layout, leaf_vx_px_s=leaf_vx)
                        last_poll_log = now
                    time.sleep(POLL_INTERVAL)
                    continue
            riding_gold = False
            if last_click == (pointer_x, zone):
                stagnation_count += 1
                if stagnation_count >= STAGNATION_LIMIT:
                    print(f"Pointer stuck at x={pointer_x} in {zone} for {stagnation_count} clicks — minigame likely over, stopping.")
                    conn.commit()
                    conn.close()
                    return
            else:
                stagnation_count = 0

            # CLAUDE.md fire-first rule: click immediately after deciding —
            # no random_delay / disk writes between sample and click, or
            # the leaf moves before we click.
            button_cx = button_region["left"] + button_region["width"] // 2
            button_cy = button_region["top"] + button_region["height"] // 2
            click(win_left + button_cx, win_top + button_cy)
            click_time = time.time()
            last_click = (pointer_x, zone)
            vx_tag = f", vx={leaf_vx:+.0f} px/s, red ahead {red_ahead}px ~{time_to_red_ms}ms" \
                if time_to_red_ms is not None else \
                (f", vx={leaf_vx:+.0f} px/s, no red ahead" if leaf_vx is not None else "")
            print(f"Pointer at x={pointer_x} in {zone} (red_d={red_dist}{vx_tag}) — chop #{chop_idx + 1}")

            if pending is not None:
                prev_row_id, prev_click_time = pending
                set_outcome(conn, prev_row_id, "survived",
                            int((click_time - prev_click_time) * 1000))

            chop_idx += 1
            row_id = log_chop(
                conn,
                session_started=session_started,
                chop_idx=chop_idx,
                clicked_at=datetime.now().isoformat(timespec="milliseconds"),
                pointer_x=pointer_x,
                zone=zone,
                nearest_red_distance=red_dist,
                red_safety_margin=RED_SAFETY_MARGIN_PX,
                bar_left=bar_region["left"],
                bar_top=bar_region["top"],
                bar_width=bar_region["width"],
                bar_height=bar_region["height"],
                button_click_x=button_cx,
                button_click_y=button_cy,
                window_w=win_w,
                window_h=win_h,
                leaf_vx_px_s=leaf_vx,
                red_ahead_px=red_ahead,
                time_to_red_ms=time_to_red_ms,
                code_commit=code_commit,
                source="bot",
            )
            log_poll(conn, session_started,
                     int((click_time - session_start_t) * 1000),
                     pointer_x, zone, red_dist, bar_px, 1,
                     zone_layout=layout,
                     leaf_vx_px_s=leaf_vx)
            last_poll_log = click_time
            pending = (row_id, click_time)

            # Random delay goes AFTER the click (no fire-time latency).
            # No blind cooldown sleep: arm the fire hold instead, so the
            # loop keeps sampling while the chop registers.
            fire_hold_layout = layout
            fire_hold_click_t = click_time
            riding_gold = False
            random_delay(20, 60)
            time.sleep(POLL_INTERVAL)
            continue

        if now - last_poll_log >= POLL_LOG_INTERVAL:
            log_poll(conn, session_started,
                     int((now - session_start_t) * 1000),
                     pointer_x, zone, red_dist, bar_px, 0,
                     zone_layout=layout,
                     leaf_vx_px_s=leaf_vx)
            last_poll_log = now

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
