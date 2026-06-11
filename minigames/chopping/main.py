import time
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay, check_failsafe
from common.regions import get_region
from common.score_ocr import read_score
from common.session_log import session_log
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from common.window import get_bounds, WindowNotFoundError
from minigames.chopping.chop_log import open_db, log_chop, log_poll, set_outcome, set_registered, set_pts
from minigames.chopping.detector import (
    analyze_bar,
    bar_pixel_count,
    eased_time_to_red_ms,
    gold_distance_ahead,
    infer_vmax,
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

# A registered chop's layout re-roll arrives within 86-201ms; a re-roll
# "seen" later than this isn't credited to the chop (it could be the
# round-end dissolve or an unrelated change). Only bounds the
# registered-flag bookkeeping, not the fire hold's release.
REROLL_ACK_MAX_S = 0.40

# Read the on-screen "N PTS" counter this long after each chop's click
# — inside the fire hold (which lasts >= MIN_INTERCHOP_S), so the
# ~100ms OCR costs no throughput, and late enough for the counter's
# increment animation to settle.
PTS_READ_AFTER_S = 0.35


def _read_pts(win_left: int, win_top: int, win_w: int, win_h: int) -> int | None:
    """OCR the live PTS counter (region "score" in regions.json, picked
    via chopping-pick-score-region). None when unpicked or unreadable."""
    region = get_region(_HERE, "score", win_w, win_h)
    if region is None:
        return None
    frame = grab_region(
        win_left + region["left"], win_top + region["top"],
        region["width"], region["height"],
    )
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
    return read_score(gray)


# Bounce-vs-chop speed experiment (2026-06-11, maintainer-designed):
# pause firing for a fixed window after every Nth REGISTERED chop and
# just record. The 01:08 analysis attributed the speed ramp to chops,
# but it had a confound: the early ramp happened while chops AND
# bounces accumulated together, and the chop-free droughts that looked
# flat were all at already-high speed where a saturated ramp would
# mask either cause. Pausing after every 5th success samples chop-free
# bounces across the WHOLE speed range — if V_max rises during a
# low-speed pause, bounces ramp it after all. The pause start/resume
# prints include the V_max estimate, so the session log itself shows
# the result; the full-rate polls record everything for
# scripts/analyze_chop_speed.py. Set EVERY_N to 0 to disable once the
# question is settled.
#
# SETTLED 2026-06-11 (10:51 session, pauses after chops 5 and 10):
# per-sweep peaks inside both chop-free pauses were FLAT at low and
# mid speed — bounces don't ramp it, chops do, now unconfounded.
# Disabled; re-enable for a high-speed (15-20 chop) confirmation run
# if ever needed.
BOUNCE_EXPERIMENT_EVERY_N = 0
BOUNCE_EXPERIMENT_PAUSE_S = 12.0

# Exit-on-starve (2026-06-11): points bank as in-game tokens on a
# voluntary exit and each game starts fresh (maintainer ground truth),
# so once the speed ramp makes safe fire windows this rare the round
# has hit its natural ceiling — end the session and bank rather than
# wait out a marginal window (that's how every bot death so far
# happened). Late-round gaps of 26-45s between chops are normal;
# 60s of no safe fire means truly starved.
STARVE_EXIT_S = 60.0

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
# margin can't represent that asymmetry.
#
# Since 2026-06-11 the time estimate is the position-aware eased model
# (detector.eased_time_to_red_ms with V_max from detector.infer_vmax)
# — the previous flat eff-speed floor (0.75 × recent-max |vx|) starved
# the bot at ~20 points: detection-jitter spikes of 1600-1800 px/s
# poisoned the recent-max for 3s at a time, pricing every window at
# ~190px of needed runway when greens are 70-110px wide. A human
# clicking greens reaches 66 points (maintainer's best), so windows
# clearly exist; the model just has to price them honestly. Budget
# 120ms: the two computable deaths come out at 47ms and 81ms under
# the eased model (validated in scripts/validate_chop_gate.py), while
# green-entry fires price at ~130-180ms — so 120 separates them with
# margin on both sides. The real decision→register latency is the
# ~40-80ms floor.
MIN_TIME_TO_RED_MS = 120

# Direction-unknown fires are banned (2026-06-11, the 00:46 death):
# the leaf read the same x twice 127ms apart (stale/frozen render),
# vx came out exactly 0.0, the old `abs(vx) > 1e-6` guard silently
# disarmed the time-to-red gate, and the click landed while the real
# leaf was doing ~355 px/s with red 31px (~87ms) ahead. Below this
# speed the direction signal is jitter/turnaround/stale — wait ~one
# poll for a clean read instead of firing blind.
MIN_VX_FOR_FIRE = 30.0

# Window of recent (x, |vx|) samples feeding detector.infer_vmax — the
# robust per-sweep peak-speed estimate for the eased time-to-red model.
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
        try:
            _run_inner()
        finally:
            # The DB is tracked so other machines get session data via
            # `git pull`. Rollback-journal mode keeps the on-disk file
            # consistent between transactions, so this is safe even if
            # _run_inner bailed before conn.close().
            commit_file_if_changed(
                _HERE.parent.parent,
                "minigames/chopping/assets/chopping.db",
                "chore(chopping): refresh chopping.db (auto)",
            )


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
    # round ended (bar disappeared). pending = (row_id, click_time, zone).
    pending: tuple[int, float, str] | None = None
    # 1 click = 1 point at human cadence (maintainer ground truth) — so
    # count the clicks the game actually accepted and estimate the score
    # for the session-end report (+1 green, +2 gold).
    registered_chops = 0
    estimated_points = 0
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
    fire_hold_rerolled = False  # layout re-roll (the in-game chop ack) seen?
    hold_pts_read = False  # PTS OCR done for the current hold?
    last_pts: int | None = None  # latest OCR'd on-screen PTS value
    riding_gold = False  # print-once flag for the gold-upgrade hold
    # Bounce-vs-chop experiment state: fire pause deadline + the V_max
    # snapshot taken at pause start (also serves as the resume flag).
    experiment_pause_until = 0.0
    experiment_v_before: float | None = None
    # (wall_time, leaf_x, |vx|) over the recent window — feeds infer_vmax.
    speed_track: list[tuple[float, int, float]] = []

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

        # In-game chop ack: the layout re-rolling shortly after a click
        # means the game accepted it as a chop. Checked at top level
        # (every poll, any leaf zone) so the ack is caught the poll it
        # happens. Late changes aren't credited — the round-end
        # dissolve also changes the layout.
        if (
            fire_hold_layout is not None
            and not fire_hold_rerolled
            and layout != fire_hold_layout
        ):
            fire_hold_rerolled = True
            if (
                pending is not None
                and now - fire_hold_click_t <= REROLL_ACK_MAX_S
                and bar_px >= BAR_DEAD_PIXEL_THRESHOLD
            ):
                set_registered(conn, pending[0], 1)
                registered_chops += 1
                estimated_points += 2 if pending[2] == "gold" else 1
                if (
                    BOUNCE_EXPERIMENT_EVERY_N
                    and registered_chops % BOUNCE_EXPERIMENT_EVERY_N == 0
                ):
                    experiment_pause_until = now + BOUNCE_EXPERIMENT_PAUSE_S
                    experiment_v_before = infer_vmax(
                        [(x, v) for _, x, v in speed_track], bar_frame.shape[1]
                    )
                    v_desc = (f"{experiment_v_before:.0f}"
                              if experiment_v_before is not None else "?")
                    print(f"  [experiment] {registered_chops} registered chops — "
                          f"pausing {BOUNCE_EXPERIMENT_PAUSE_S:.0f}s to record "
                          f"chop-free bounces (V_max now {v_desc} px/s)")
        if bar_px < BAR_DEAD_PIXEL_THRESHOLD:
            if bar_dead_since is None:
                bar_dead_since = now
            elif now - bar_dead_since >= BAR_DEAD_GRACE_S:
                if pending is not None:
                    row_id, click_time, _ = pending
                    set_outcome(conn, row_id, "round_ended",
                                int((now - click_time) * 1000))
                    pending = None
                print(f"Bar gone for {now - bar_dead_since:.1f}s (bar_px={bar_px}) — round ended, stopping.")
                print(f"Session result: {chop_idx} clicks, {registered_chops} registered "
                      f"by the game, ~{estimated_points} points expected, last in-game "
                      f"PTS read: {last_pts if last_pts is not None else 'unread'}.")
                conn.commit()
                conn.close()
                return
        else:
            bar_dead_since = None

        # Exit-on-starve: see STARVE_EXIT_S. The last chop's outcome is
        # 'survived' — nothing died; the round just priced the bot out.
        if chop_idx > 0 and now - fire_hold_click_t > STARVE_EXIT_S:
            if pending is not None:
                row_id, click_time, _ = pending
                set_outcome(conn, row_id, "survived",
                            int((now - click_time) * 1000))
                pending = None
            print(f"No safe fire window for {STARVE_EXIT_S:.0f}s — round has hit "
                  f"its ceiling, stopping to bank the tokens.")
            final_pts = _read_pts(win_left, win_top, win_w, win_h) or last_pts
            print(f"Session result: {chop_idx} clicks, {registered_chops} registered "
                  f"by the game, ~{estimated_points} points expected, "
                  f"in-game PTS: {final_pts if final_pts is not None else 'unread'}. "
                  f"EXIT the minigame in-game to bank them (each game starts fresh).")
            conn.commit()
            conn.close()
            return

        red_dist = None
        if pointer_x is None:
            leaf_track.clear()
        else:
            red_dist = nearest_red_distance(bar_frame, pointer_x)
            leaf_track.append((now, pointer_x))
            while leaf_track and now - leaf_track[0][0] > 0.3:
                leaf_track.pop(0)
        leaf_vx = leaf_vx_px_s(leaf_track)
        if leaf_vx is not None and pointer_x is not None:
            speed_track.append((now, pointer_x, abs(leaf_vx)))
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
                interchop_ok = now - fire_hold_click_t >= MIN_INTERCHOP_S
                fallback = now - fire_hold_click_t >= COOLDOWN_AFTER_CLICK
                if (fire_hold_rerolled and interchop_ok) or fallback:
                    if fallback and not fire_hold_rerolled and pending is not None:
                        # No ack ever arrived — the game ignored the click.
                        set_registered(conn, pending[0], 0)
                    fire_hold_layout = None
                else:
                    # PTS ground truth: one OCR per hold, after the
                    # counter's increment settles. Sits inside the
                    # mandatory inter-chop wait — zero throughput cost.
                    if not hold_pts_read and now - fire_hold_click_t >= PTS_READ_AFTER_S:
                        hold_pts_read = True
                        pts = _read_pts(win_left, win_top, win_w, win_h)
                        if pts is not None:
                            last_pts = pts
                            if pending is not None:
                                set_pts(conn, pending[0], pts)
                    if now - last_poll_log >= POLL_LOG_INTERVAL:
                        log_poll(conn, session_started,
                                 int((now - session_start_t) * 1000),
                                 pointer_x, zone, red_dist, bar_px, 0,
                                 zone_layout=layout, leaf_vx_px_s=leaf_vx)
                        last_poll_log = now
                    time.sleep(POLL_INTERVAL)
                    continue

            # Bounce-vs-chop experiment: deliberate fire pause (see
            # BOUNCE_EXPERIMENT_EVERY_N). Polls keep recording at full
            # rate throughout — that's the experiment.
            if now < experiment_pause_until:
                if now - last_poll_log >= POLL_LOG_INTERVAL:
                    log_poll(conn, session_started,
                             int((now - session_start_t) * 1000),
                             pointer_x, zone, red_dist, bar_px, 0,
                             zone_layout=layout, leaf_vx_px_s=leaf_vx)
                    last_poll_log = now
                time.sleep(POLL_INTERVAL)
                continue
            if experiment_pause_until and now >= experiment_pause_until:
                v_after = infer_vmax(
                    [(x, v) for _, x, v in speed_track], bar_frame.shape[1]
                )
                before = (f"{experiment_v_before:.0f}"
                          if experiment_v_before is not None else "?")
                after = f"{v_after:.0f}" if v_after is not None else "?"
                print(f"  [experiment] resuming after the chop-free pause: "
                      f"V_max {before} -> {after} px/s "
                      f"(rise = bounces ramp it; flat = chops do)")
                experiment_pause_until = 0.0
                experiment_v_before = None

            # Time-to-red gate (directional, the load-bearing one) plus
            # the legacy pixel-margin floor. Direction must be KNOWN
            # (see MIN_VX_FOR_FIRE); the time estimate is the
            # position-aware eased model (see MIN_TIME_TO_RED_MS).
            red_ahead = None
            time_to_red_ms = None
            direction_known = leaf_vx is not None and abs(leaf_vx) >= MIN_VX_FOR_FIRE
            if direction_known:
                red_ahead = red_distance_ahead(bar_frame, pointer_x, leaf_vx)
                if red_ahead is not None:
                    v_max = infer_vmax(
                        [(x, v) for _, x, v in speed_track], bar_frame.shape[1]
                    )
                    if v_max is not None:
                        time_to_red_ms = eased_time_to_red_ms(
                            pointer_x, leaf_vx, red_ahead,
                            max(v_max, abs(leaf_vx)), bar_frame.shape[1],
                        )
            unsafe = (
                not direction_known
                or (red_dist is not None and red_dist < RED_SAFETY_MARGIN_PX)
                # Red ahead but V_max not yet estimable (first ~150ms of
                # a round): wait for the estimate rather than fire blind.
                or (red_ahead is not None and time_to_red_ms is None)
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
                    print(f"Session result: {chop_idx} clicks, {registered_chops} registered "
                          f"by the game, ~{estimated_points} points expected.")
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
                prev_row_id, prev_click_time, _ = pending
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
            pending = (row_id, click_time, zone)

            # Random delay goes AFTER the click (no fire-time latency).
            # No blind cooldown sleep: arm the fire hold instead, so the
            # loop keeps sampling while the chop registers.
            fire_hold_layout = layout
            fire_hold_click_t = click_time
            fire_hold_rerolled = False
            hold_pts_read = False
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
