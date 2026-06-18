"""Fishing minigame bot — main loop and config.

The fishing minigame is a hold-to-charge distance game (see
docs/fishing_minigame.md): hold LMB to fill a charge bar, release to land the
lure on a fish (Green/Eel/Squid/Whale) while avoiding mines. The bot casts
CLOSED-LOOP — it polls the charge bar while holding and releases at a target
fill (common.input.charge_and_release) — so the charge LEVEL is the control
variable; its learned model is charge_level <-> cast distance (cast_model),
fitted at startup from logged casts and refined by exploration (darts pattern).

STATUS: auto-starts (clicks the PLAY GAME prompt, like catching/mining — the
cast bar is the 'minigame active' signal), plays the picked cast-bar region
with calibrated colour detection, and aims off the charge bar. The charge fill
is a clean, in-crop, reproducible distance signal (validated offline on the
run-7 frames: landed_x ~ 5.0*charge); the closed-loop release also detects the
rod-not-ready state (fill stays 0 = previous lure still reeling) and skips that
cast instead of burning it (#58). Until MIN_SAMPLES charged casts exist the bot
explores random target charge levels and logs (charge_level, landed_dist).

Catches are detected by the PTS score DELTA (read before vs after each cast —
the game's own count, authoritative for the catch + points), which overrides and
rescues the far-cast / eel catches the bobber-disappearance heuristic misses
(#58/#63). The delta is the TOTAL points: sliding fish converge and are caught
together, so it can be a SUM (+1 green / +2 eel / +3 squid as a lone fish; any
other positive delta is a 'multi' catch worth its delta — NOT a whale). The score read falls back to that heuristic whenever
a digit isn't captured yet / the crop is unreadable (read is None). Grow the
digit-template library with `fishing --save-frames` + `fishing-capture-digits`.

The targets SLIDE (~cast 5+), so for a single-fish model cast the bot samples the
fish's velocity over a few quick polls and LEADS the aim to its predicted landing
(cast_model.lead_fish_dist) — shipped as a parity A/B (aim_mode model_lead vs
model_nolead) to prove it before defaulting, since the catch radius is tight
(#58). Open: warm-fish HSV verification + grey-mine detection (#63).
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.monitor import save_frame
from common.input import charge_and_release, click, random_delay, check_failsafe
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from common.git_info import current_code_commit
from common.auto_commit import commit_file_if_changed
from minigames.fishing.detector import (
    find_fish, find_mines, find_lure, find_game_over, find_cast_bar,
    find_play_button, find_charge_fill, find_eel,
)
from minigames.fishing.fish_log import (
    open_db, log_cast, set_outcome, log_run, fetch_cast_samples,
)
from minigames.fishing.cast_model import (
    FISH_VALUE, MIN_SAMPLES, fit_cast_model, charge_for_distance,
    distance_for_charge, choose_target, lands_on_mine_only, nearest_fish_target,
    theil_sen_velocity, lead_fish_dist, LEAD_TIME_FALLBACK_S,
)
from minigames.fishing.score import read_score, kind_from_delta, score_crop

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
FISH_DB_PATH = _HERE / "assets" / "fishing.db"
REPO_ROOT = _HERE.parent.parent

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.05

# Exploration charge range (fill px): random target charge levels within this
# span sample the charge->distance curve until the cast model is fitted.
# The charge thermometer ramps 0->~56 over the up-sweep then turns DOWN (it's a
# release-timing oscillation, #58). Keep explore targets safely below the peak
# so they're always reached on the up-sweep — a target above the peak would
# never trigger the release and the cast would fall to the max-hold backstop.
# The MIN is low (5, not 10) so exploration samples SHORT casts and the model's
# reach floor drops toward fish that spawn close to the dock — otherwise they sit
# below reach (~69px at charge 11) and can't be targeted (#58). It doubles as the
# floor for the close-fish 'nearest' aim (charge_for_distance min_charge).
EXPLORE_CHARGE_MIN = 5
EXPLORE_CHARGE_MAX = 50
# Fire an exploration cast every Nth cast even after the model is fitted, so
# the charge->distance surface keeps getting sampled (darts EXPLORE_EVERY_N).
EXPLORE_EVERY_N = 10

# Closed-loop charge-and-release timing (common.input.charge_and_release):
# poll the charge bar this often while holding; give the rod this long to
# start charging before declaring it not-ready (previous lure still reeling);
# never hold longer than this (the bar saturates ~750ms, so this bounds a
# stuck hold and the wait on an over-cap target).
CHARGE_POLL_S = 0.012  # tight poll: the bar rises ~0.1px/ms, so release-overshoot ~ grab(~13ms)+poll
# With the thermometer read correctly (find_charge_fill), the fill rises from 0
# within ~2-3 polls of a real hold, so the not-ready grace can be short; a
# genuinely-reeling rod leaves it at 0 and aborts here. (The old 0.8s was a
# band-aid for the wrong reading — x<10 read 0 forever, runs 8-10, #58.)
CHARGE_READY_GRACE_S = 0.4
# Backstop for a target the up-sweep never reaches: release near the up-sweep
# PEAK (~780ms) rather than holding into the down-sweep, which would cast at an
# unpredictable (low) fill. So cap just past the peak.
CHARGE_MAX_HOLD_S = 0.95
# After a not-ready abort, wait this long before retrying (the rod recovers
# over ~a cast cycle; the aborts are cheap — they poll for ready without
# burning a cast, unlike the old fixed-cooldown open-loop hold).
NOT_READY_RETRY_S = 0.3

# Landing measurement: the lure REELS IN after landing (the bobber's x
# decreases poll-to-poll until it vanishes — confirmed in the --save-frames
# trajectories, e.g. 161->126->62->15->gone). So the landing is the FARTHEST x
# the bobber reaches (max x), NOT the last/settled read (that's the retraction
# endpoint — the bug behind the same hold "landing" at 256 and 27). Poll,
# track the max x, and early-exit once the bobber retracts past it
# (x < max - LANDING_RETRACT_PX) or vanishes after being seen — keeping the
# grab count (screen-capture lag) low. CAST_SETTLE_S caps the wait (the lure
# arcs high / off-crop before landing on long casts).
CAST_SETTLE_S = 1.6
LANDING_POLL_S = 0.1
LANDING_RETRACT_PX = 25   # x dropped this far below the max = reeling in (landed)
# A landing is trusted only if the lure SETTLED there: >= MIN_LANDING_DETECTIONS
# sightings within LANDING_STABLE_PX of the farthest x. A lone find_lure flicker
# (a mid-arc catch or a false match — e.g. run-13 cast 2's single x=93) was being
# logged as a landing and corrupting the fill->distance fit (#58).
LANDING_STABLE_PX = 10
MIN_LANDING_DETECTIONS = 2
# A catch is a fish that sat within this many px of the landing just before the
# lure arrived and is GONE after (the caught fish vanishes). Observed catch
# offsets: ~2-11px (run 14 casts 6/9), so 15 covers a hit while excluding a
# fish the lure clearly overshot (cast 3: fish 203, lure 226 = miss).
CATCH_RADIUS = 15
# Max px a fish can SLIDE between the pre-arrival frame and the landed frame, used
# by vanished_fish to PAIR a near-landing fish to a post sighting of the same kind
# (it moved, wasn't caught) vs counting it caught. Set to the documented MAX
# per-cast slide (~32px over the full ~1.3s decision->landing; pre->post is only
# the lure's visible flight, so this is a generous upper bound). The asymmetry
# justifies erring large: over-pairing only loses a multi-attribution to the SAFE
# fallback (the score-delta kind guess + a +1 streak stand), while UNDER-pairing
# counts a slid-away fish as caught — the kind-only bug this fixes (#66, review).
# Calibratable from --save-frames pre/post pairs.
SLIDE_BUDGET_PX = 32
# Moving-target lead: the targets SLIDE back and forth (~cast 5+, growing
# through a game; see docs). For a single-fish model cast the bot samples the
# fish over a few quick polls BEFORE firing to estimate its slide velocity
# (Theil-Sen), then leads the aim to the predicted landing — shipped as a parity
# A/B (aim_mode model_lead vs model_nolead, cast_model.lead_fish_dist). The
# samples come from frames grabbed before the cast decision, so the closed-loop
# cast isn't biased by the small added latency and no disk write goes between the
# decision and the hold. Both arms sample (matched latency); only the lead arm
# applies the lead.
LEAD_SAMPLES = 4            # fish sightings for the velocity fit (incl. the decision frame)
LEAD_SAMPLE_DT_S = 0.05     # spacing between velocity polls (span ~0.15s, above the noise floor)

# Wait between casts (the next charge can't start until the lure resets).
CAST_COOLDOWN_S = 0.6
# Auto-start (like catching/mining): click the PLAY GAME prompt to begin ONE
# minigame, play it, then STOP on game-over (no auto-replay). After a click,
# wait this long before re-clicking (the minigame is loading).
START_CLICK_COOLDOWN_S = 2.0
# Give up after this many PLAY GAME clicks that don't produce a cast bar.
MAX_START_ATTEMPTS = 3
# Once playing, the cast bar being gone this long means the attempt ended
# (rides out brief detection dropouts). The cast bar is the 'active' signal —
# it only exists during play, unlike a fish/colour count that the entry scene
# would false-trigger. Game-over template detection (game_over.png) is primary
# when that asset exists.
BAR_GONE_GAMEOVER_S = 3.0


def _measure_landing(win_left, win_top, play, settle_s, poll_s,
                     frames_dir=None, cast_idx=0):
    """Poll find_lure after a cast and return (landed (x, y), frame): the
    bobber's FARTHEST x = the landing (it reels in after landing, so max-x is
    the landing, not the settled/retracted read). Exits early once the bobber
    retracts past the max or vanishes after being seen, keeping the grab count
    low. (None, None) if never seen (cast landed off the play region).

    The cast power is the charge bar (returned by charge_and_release at
    release), not measured here — this poll only locates the bobber to classify
    the hit and supply landed_dist as the model's training target.

    The landing is trusted only if the lure SETTLED (MIN_LANDING_DETECTIONS
    sightings near the farthest x); a single flicker returns (None, None, ...) so
    it isn't logged (#58).

    Returns (landing (x, y), landed_frame, pre_frame). `pre_frame` is the last
    frame BEFORE the lure first appears — the catch test compares the fish there
    (about to be caught) against the landed frame, since a caught fish DISAPPEARS
    (a fish present at the landing post-cast means it was NOT caught, #58).

    When frames_dir is set (--save-frames), every poll frame is saved with the
    find_lure x (or NA) in the name — for debugging which moment the landing
    read latches onto (#58)."""
    dets: list[tuple[int, int, "object"]] = []   # all (x, y, frame) sightings
    pre_frame = None             # most recent frame before the lure appears
    seen = False
    deadline = time.time() + settle_s
    j = 0
    while time.time() < deadline:
        post = grab_region(
            win_left + play["left"], win_top + play["top"],
            play["width"], play["height"],
        )
        lure = find_lure(post)
        if frames_dir is not None:
            lx = lure[0] if lure is not None else "NA"
            save_frame(frames_dir / f"cast{cast_idx:02d}_p{j:02d}_x{lx}.png", post)
        j += 1
        if lure is not None:
            seen = True
            dets.append((lure[0], lure[1], post))
            if lure[0] < max(d[0] for d in dets) - LANDING_RETRACT_PX:
                break              # reeling in past the landing -> done
        elif seen:
            break                  # bobber vanished after landing -> done
        else:
            pre_frame = post       # not seen yet -> keep as the pre-catch frame
        time.sleep(poll_s)
    landing = _landing_from_detections(dets)
    if landing is None:
        return None, None, pre_frame
    return (landing[0], landing[1]), landing[2], pre_frame


def _classify_catch(pre_fish: list[dict], post_fish: list[dict], landed_x: int,
                    radius: int = CATCH_RADIUS) -> tuple[str, int]:
    """Classify a cast outcome from the fish near the landing BEFORE vs AFTER the
    lure arrives. A catch = a fish that was within `radius` of the landing just
    before is GONE after (the caught fish is consumed/vanishes); a fish still
    sitting there means the lure missed it. Returns (kind, made).

    This replaces the old 'is a fish at the landing now?' test, which scored
    every catch as a miss because the fish has already vanished by then (#58)."""
    near = sorted((f for f in pre_fish if abs(f["x"] - landed_x) <= radius),
                  key=lambda f: abs(f["x"] - landed_x))
    if not near:
        return "miss", 0          # nothing was there to catch
    if any(abs(f["x"] - landed_x) <= radius for f in post_fish):
        return "miss", 0          # a fish is still there -> not caught
    return near[0]["kind"], 1     # the nearest fish vanished -> caught


def vanished_fish(pre_fish: list[dict], post_fish: list[dict], landed_x: int,
                  radius: int = CATCH_RADIUS,
                  slide_budget: int = SLIDE_BUDGET_PX) -> list[dict]:
    """Fish that were near the landing just before the lure arrived and are GONE
    after — the actual catch(es). The sliding mechanic converges several fish, so
    one cast can consume more than one; this returns a LIST (vs _classify_catch's
    single nearest). Pure.

    Matching is by kind AND POSITION: each near-landing pre fish is paired to the
    nearest still-unclaimed post fish of the SAME kind within `slide_budget` px (it
    moved, wasn't caught); a pre fish with no such match VANISHED -> caught. The
    position pairing (not kind-alone) is what stops a fish that merely SLID out of
    the landing radius from being miscounted as caught — fish routinely slide
    >radius between frames (docs), so kind-alone over-counts (#66, review-flagged).

    A caught fish that was UNDETECTED in `pre_fish` (e.g. an under-matched eel)
    simply isn't in the result — `attribute_catch` then cross-checks the vanished
    set's value-sum against the score delta and falls back to the delta's kind
    guess on any mismatch, so this can stay permissive. The rare residual it can't
    catch (an undetected catch whose value is offset by a flickered-out uncaught
    fish of equal value) only mis-labels data; points/made/streak-gating are
    unaffected."""
    near_pre = sorted((f for f in pre_fish if abs(f["x"] - landed_x) <= radius),
                      key=lambda f: abs(f["x"] - landed_x))
    remaining = list(post_fish)
    vanished: list[dict] = []
    for f in near_pre:
        match_i = match_d = None
        for i, p in enumerate(remaining):
            if p["kind"] != f["kind"]:
                continue
            d = abs(p["x"] - f["x"])
            if d <= slide_budget and (match_d is None or d < match_d):
                match_i, match_d = i, d
        if match_i is None:
            vanished.append(f)          # no same-kind post sighting nearby -> caught
        else:
            remaining.pop(match_i)      # paired to a post sighting -> slid, not caught
    return vanished


def attribute_catch(vanished: list[dict], delta: int | None
                    ) -> tuple[str | None, int | None]:
    """Name a measured catch's kind(s) and fish COUNT from the vanished set, but
    ONLY when their FISH_VALUE sum equals the score `delta` — the cross-check that
    rejects a detection flicker (a dropped/extra fish makes the sum mismatch).
    Returns ``(kind_label, n_fish)`` — a '+'-joined sorted label ('eel+squid', or
    just 'green' for a lone fish) and the fish count — or ``(None, None)`` when the
    set can't be trusted (nothing vanished, no positive delta, or the sum
    mismatches), so the caller keeps the score-delta's kind guess and a +1 streak.

    This disambiguates what the delta alone can't: a +2 is a lone eel OR two
    greens, a +5 ('multi') is eel+squid — naming the kinds on measured catches and
    giving the per-fish count the streak undercounts on converged catches
    (#66/#70). Far casts with no measured bobber pass `vanished=[]` and fall back.
    Pure.

    Only SCORING fish (FISH_VALUE > 0) count toward the label, sum, and count: a
    zero-value fish (megalodon, or any future non-scoring kind) must never inflate
    the streak count nor let the value-sum match coincidentally. (Today's detector
    leaves megalodon off, so the vanished set is already all-scoring — this keeps
    the guard from depending on that.)"""
    scoring = [f for f in vanished if FISH_VALUE.get(f["kind"], 0) > 0]
    if not scoring or not delta or delta <= 0:
        return None, None
    if sum(FISH_VALUE[f["kind"]] for f in scoring) != delta:
        return None, None
    return "+".join(sorted(f["kind"] for f in scoring)), len(scoring)


def _streak_after(streak: int, landed_kind: str | None, n_caught: int) -> int:
    """The streak after a catch. A Whale resets it to 1 (wiki) — tested against
    the '+'-split label so a CONVERGED catch containing a whale ('green+whale')
    resets too, not only the lone 'whale' (review-flagged). Otherwise the streak
    advances by the NUMBER OF FISH caught (the game counts per fish, so a converged
    multi-catch is +2/+3); n_caught is 1 when the count can't be measured (#70)."""
    if "whale" in (landed_kind or "").split("+"):
        return 1
    return streak + n_caught


def _resolve_catch(heuristic: tuple[str | None, int | None, str | None],
                   score_before: int | None, score_after: int | None
                   ) -> tuple[str | None, int | None, str | None]:
    """Final (landed_kind, made, detect_source) for a cast.

    The PTS score DELTA is the game's own count, used to rescue / correct the
    bobber-disappearance / eel-absence heuristic (#58/#63). `heuristic` is the
    (kind, made, source) detection produced (made None when the cast wasn't
    measured). The two reads are trusted ASYMMETRICALLY:

    - delta > 0 (a known fish value): AUTHORITATIVE catch — the score rose, so
      something was caught; this is the rescue for far-cast / eel catches that
      never settle a measurable bobber.
    - delta == 0 (a clean miss): the counter can lag the catch by an animation
      frame, so a zero read right after the landing may be STALE. Don't let it
      veto a catch the bobber/eel heuristic positively confirmed (that would
      fabricate a miss and reset the streak — the dangerous failure). Trust the
      zero as a miss only when the heuristic did NOT confirm a catch.
    - a None read (uncaptured digit / unreadable) or an ambiguous delta leaves
      the heuristic verdict in place.
    """
    _h_kind, h_made, _h_source = heuristic
    delta_catch = kind_from_delta(score_before, score_after)
    if delta_catch is None:
        return heuristic
    kind, _delta = delta_catch
    if kind != "miss":
        return (kind, 1, "score")          # delta>0: the score rose -> caught
    if h_made == 1:
        return heuristic                   # don't let a stale zero veto a catch
    return ("miss", 0, "score")            # zero confirms / records the miss


def _landing_from_detections(dets, stable_px: int = LANDING_STABLE_PX,
                             min_dets: int = MIN_LANDING_DETECTIONS):
    """The landing from a list of (x, y, frame) lure sightings: the entry at the
    FARTHEST x, accepted only if at least `min_dets` sightings settled within
    `stable_px` of it. Returns that (x, y, frame), or None when the lure was seen
    too few times / only flickered (so the cast isn't logged with a bad landing,
    #58). Pure — the IO poll lives in _measure_landing."""
    if len(dets) < min_dets:
        return None
    farthest = max(dets, key=lambda d: d[0])
    near = [d for d in dets if d[0] >= farthest[0] - stable_px]
    if len(near) < min_dets:
        return None              # the max was a lone transient, not a settle
    return farthest


def _cast_origin(play_region: dict, bar: tuple[int, int, int, int] | None) -> tuple[int, int]:
    """Lure cast origin (play-region-relative) — distance is measured from
    here, so it anchors the cast model. The lure is cast RIGHTWARD along the
    cast bar, so the origin is the bar's LEFT edge at its mid-height (the track
    start near the player). Falls back to the play-region centre when the bar
    isn't detected."""
    if bar is not None:
        x, y, w, h = bar
        return (x, y + h // 2)
    return (play_region["width"] // 2, int(play_region["height"] * 0.66))


# Dynamic play-crop padding around the DETECTED cast bar (full-window coords).
# The play crop is anchored to the bar — NOT a fixed regions.json rectangle — so
# it follows the player-anchored minigame. A fixed crop barely contained the bar
# (2px of margin) and clipped its left edge off-screen when the player stood a
# little differently, so the charge bar / score / cast origin anchored to that
# edge all read garbage and the whole session failed (#58). Left pad clears the
# cast origin (the bar's left edge); the vertical pad covers the fish that sit
# above/below the bar (detector.BAR_VPAD).
PLAY_PAD_L = 55
PLAY_PAD_R = 30
PLAY_PAD_V = 30


def _play_crop_from_bar(bar_full, win_w, win_h,
                        pad_l=PLAY_PAD_L, pad_r=PLAY_PAD_R, pad_v=PLAY_PAD_V):
    """A dynamic play-crop rect (window-relative ``{left, top, width, height}``)
    around the cast bar detected on the FULL window, plus the bar's position
    WITHIN that crop. Anchoring the crop to the bar (clamped to the window) keeps
    detection player-position-independent. Pure."""
    bx, by, bw, bh = bar_full
    pl, pt = max(0, bx - pad_l), max(0, by - pad_v)
    pr, pb = min(win_w, bx + bw + pad_r), min(win_h, by + bh + pad_v)
    play = {"left": pl, "top": pt, "width": pr - pl, "height": pb - pt}
    bar = (bx - pl, by - pt, bw, bh)
    return play, bar


def _explore_charge(model, origin_x, mines, fish, tries=6):
    """A random exploration charge whose predicted landing avoids a mine-only
    spot (wiki: a mine with no fish fails; a fish even over a mine scores).
    Model casts target a fish and are already safe, so this only guards the
    explore/fallback casts that fire at no particular fish. With no model the
    landing can't be predicted (just return a random charge); falls back to the
    last draw if every try lands on a mine."""
    c = random.randint(EXPLORE_CHARGE_MIN, EXPLORE_CHARGE_MAX)
    if model is None or not mines:
        return c
    for _ in range(tries):
        landing_x = origin_x + distance_for_charge(model, c)
        if not lands_on_mine_only(landing_x, mines, fish):
            return c
        c = random.randint(EXPLORE_CHARGE_MIN, EXPLORE_CHARGE_MAX)
    return c


def _sample_fish_velocity(win_left, win_top, play, bar, kind, seed_x,
                          n=LEAD_SAMPLES, dt=LEAD_SAMPLE_DT_S):
    """Track one fish over a few quick polls to estimate its slide velocity for
    the moving-target lead. Returns ``([(monotonic_t, x), ...], latest_x)``: the
    first sample is seeded from the caller's decision-frame detection, then n-1
    more are taken `dt` apart. Stops early if the fish drops out — a gap would
    fake a large velocity, so the buffer is kept contiguous. IO only; the robust
    velocity fit is cast_model.theil_sen_velocity. Used for both A/B arms (matched
    latency); the samples are grabbed BEFORE the cast decision, so the closed-loop
    cast isn't biased and nothing is written between the decision and the hold."""
    samples = [(time.monotonic(), float(seed_x))]
    latest = seed_x
    for _ in range(n - 1):
        time.sleep(dt)
        crop = grab_region(win_left + play["left"], win_top + play["top"],
                           play["width"], play["height"])
        same = [f for f in find_fish(crop, bar=bar) if f["kind"] == kind]
        if not same:
            break              # dropped the fish -> keep the buffer contiguous
        latest = min(same, key=lambda f: abs(f["x"] - latest))["x"]
        samples.append((time.monotonic(), float(latest)))
    return samples, latest


def run():
    parser = argparse.ArgumentParser(description="Fishing minigame bot")
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save each cast's landing-poll frames to assets/captures/"
             "botrun_<stamp>/ (gitignored), named with the find_lure x — to "
             "debug landing detection (the launcher's toggle sets "
             "FISHING_SAVE_FRAMES).")
    args = parser.parse_args()
    save_frames = args.save_frames or os.environ.get(
        "FISHING_SAVE_FRAMES", "").strip().lower() in ("1", "on", "true", "yes")
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        session_started = datetime.now().isoformat(timespec="seconds")
        code_commit = current_code_commit(REPO_ROOT)
        if code_commit:
            print(f"Code commit: {code_commit}")
        db = open_db(FISH_DB_PATH)
        # Fit the hold<->distance cast model from past measured casts (darts
        # startup-fit pattern). None until MIN_SAMPLES landings exist, which
        # needs assets/lure.png for landing measurement — until then the bot
        # explores (random holds) and logs them.
        samples = fetch_cast_samples(db)
        model = fit_cast_model(samples)
        if model is None:
            print(f"cast model: not fitted ({len(samples)} charged casts "
                  f"< {MIN_SAMPLES}) — exploring charge levels "
                  f"[{EXPLORE_CHARGE_MIN},{EXPLORE_CHARGE_MAX}]")
        else:
            lo, hi = model.reach_px()
            print(f"cast model: fitted on {model.n} casts — "
                  f"charge {model.charge_min}-{model.charge_max} reaches "
                  f"{lo:.0f}-{hi:.0f} px")
        started_at = time.time()
        stats = {"n_casts": 0, "n_not_ready": 0, "points_total": 0,
                 "max_streak": 0, "end_reason": "process_exit"}
        try:
            _run_inner(session_started, db, code_commit, model, stats,
                       save_frames=save_frames)
        except KeyboardInterrupt:
            stats["end_reason"] = "keyboard_interrupt"
            raise
        except Exception as e:  # FailSafeException (corner-slam) and any crash
            stats["end_reason"] = type(e).__name__
            raise
        finally:
            log_run(
                db,
                session_started=session_started,
                attempt_idx=1,
                ended_at=datetime.now().isoformat(timespec="seconds"),
                n_casts=stats["n_casts"],
                n_not_ready=stats["n_not_ready"],
                points_total=stats["points_total"] or None,
                max_streak=stats["max_streak"],
                duration_s=round(time.time() - started_at, 1),
                end_reason=stats["end_reason"],
                code_commit=code_commit,
            )
            db.close()
            commit_file_if_changed(
                REPO_ROOT,
                "minigames/fishing/assets/fishing.db",
                "chore(fishing): refresh fishing.db (auto)",
            )


def _run_inner(session_started, db, code_commit, model, stats, save_frames=False):
    print(f"Fishing bot starting — tracking window {WINDOW_TITLE!r}. "
          f"Move mouse to a corner to abort.")
    time.sleep(2)

    frames_dir = None
    if save_frames:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frames_dir = _HERE / "assets" / "captures" / f"botrun_{stamp}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving cast frames to {frames_dir} (--save-frames)")

    game_running = False        # confirmed active: the cast bar is present
    start_attempts = 0          # consecutive PLAY GAME clicks with no game started
    last_start_click = 0.0
    last_active_time = time.time()   # wall-clock the cast bar was last seen
    streak = 0
    charge_attempt = 0   # every charge_and_release call (ready or not) — frame tag
    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        # Detect the cast bar on the FULL window (not a fixed regions.json crop,
        # which clipped the bar's left edge off-screen when the player stood a bit
        # differently — breaking the charge/score/origin anchored to it, #58). The
        # bar is the 'minigame active' signal: it only exists during play (a
        # colour count would false-trigger on the entry scene).
        full = grab_region(win_left, win_top, win_w, win_h)
        bar_full = find_cast_bar(full)

        if bar_full is None:
            if not game_running:
                # --- Startup: click the PLAY GAME prompt (like catching/mining).
                # The prompt sits above the player; match it on the full window.
                btn = find_play_button(full)
                if btn is not None:
                    if time.time() - last_start_click < START_CLICK_COOLDOWN_S:
                        time.sleep(POLL_INTERVAL)
                        continue
                    if start_attempts >= MAX_START_ATTEMPTS:
                        stats["end_reason"] = "no_start"
                        print(f"Clicked PLAY GAME {start_attempts}x with no minigame "
                              f"starting — giving up.")
                        return
                    bx, by = btn
                    start_attempts += 1
                    last_start_click = time.time()
                    print(f"PLAY GAME at ({bx},{by}) — starting minigame "
                          f"(attempt {start_attempts}/{MAX_START_ATTEMPTS}).")
                    click(win_left + bx, win_top + by)
                    continue
                # No prompt and no bar yet — the minigame is loading; wait.
                time.sleep(POLL_INTERVAL)
                continue
            # Was playing and the bar vanished -> the attempt ended. STOP (one
            # game per run, like catching — no auto-replay).
            if time.time() - last_active_time > BAR_GONE_GAMEOVER_S:
                stats["end_reason"] = "game_over"
                print(f"Cast bar gone {BAR_GONE_GAMEOVER_S:.0f}s — minigame over. "
                      f"Casts={stats['n_casts']} points={stats['points_total']}.")
                return
            time.sleep(POLL_INTERVAL)
            continue

        # --- Cast bar present: the minigame is active. Anchor a dynamic play
        # crop to the detected bar so everything downstream (fish, origin, score,
        # charge, landing) is player-position-independent. `frame` is a slice of
        # the full window; `bar` is the bar's position within it. ---
        play, bar = _play_crop_from_bar(bar_full, win_w, win_h)
        frame = full[play["top"]:play["top"] + play["height"],
                     play["left"]:play["left"] + play["width"]].copy()
        if not game_running:
            game_running = True
            print(f"Minigame active (cast bar at {bar}) — casting.")
        last_active_time = time.time()

        is_over, go_conf = find_game_over(frame)
        if is_over:
            stats["end_reason"] = "game_over"
            print(f"Game over detected (conf={go_conf:.2f}). "
                  f"Casts={stats['n_casts']} points={stats['points_total']}.")
            return

        # Confine detection to the cast bar — over the raw frame the world
        # scenery (tan dock, shore plants) floods the colour masks (#63).
        mines = find_mines(frame, bar=bar)
        fish = find_fish(frame, bar=bar, mines=mines)
        if not fish:
            # Bar present but no fish this poll (detection gap / between spawns).
            time.sleep(POLL_INTERVAL)
            continue

        origin_x, origin_y = _cast_origin(play, bar)
        explore = model is None or (stats["n_casts"] + 1) % EXPLORE_EVERY_N == 0

        # Moving-target lead telemetry (defaults = no lead).
        lead_vx = lead_time_s = None
        lead_px_int = lead_px_eff = 0.0
        lead_clamped = lead_n = 0

        if explore:
            target_charge = _explore_charge(model, origin_x, mines, fish)
            target, aim_mode = None, "explore"
        else:
            target = choose_target(fish, model, origin_x)
            if target is None:
                # All detected fish are out of the model's reach — typically too
                # CLOSE to the dock (below the ~69px reach floor). Aim SHORT at the
                # nearest one, extrapolating the charge down to EXPLORE_CHARGE_MIN,
                # rather than a random far explore that flings the lure past them
                # into a mine (#58). The cast lands near the dock either way, so
                # it's safe even where the linear fit drifts at low charge; the
                # score delta records a catch if it lands. (Exploration's lowered
                # MIN extends the reach floor down so these become true model casts.)
                target = nearest_fish_target(fish, origin_x)
                target_charge = charge_for_distance(
                    model, target["target_dist"], min_charge=EXPLORE_CHARGE_MIN)
                aim_mode = "nearest"
            else:
                # The targets slide, so the decision-time x is stale by landing.
                # For a SINGLE fish, sample its velocity over a few quick polls
                # and refresh the target x (both A/B arms — matched latency).
                if len(fish) == 1:
                    samples, latest = _sample_fish_velocity(
                        win_left, win_top, play, bar, target["kind"], target["x"])
                    lead_n = len(samples)
                    lead_vx = theil_sen_velocity(samples)
                    target = {**target, "x": latest,
                              "target_dist": abs(latest - origin_x)}
                base_dist = target["target_dist"]
                # Parity A/B: lead the slide on even casts (model_lead), hold the
                # un-led aim on odd (model_nolead), so the two accumulate under
                # matched conditions — promote off the A/B only on disjoint-CI
                # evidence (#48). Only the lead arm applies the velocity.
                lead_arm = stats["n_casts"] % 2 == 0
                if lead_arm and lead_vx is not None and model is not None:
                    lead_time_s = LEAD_TIME_FALLBACK_S
                    lo, hi = model.reach_px()
                    led_dist, lead_px_eff, clamped = lead_fish_dist(
                        base_dist, lead_vx, lead_time_s, lo, hi)
                    lead_px_int = lead_vx * lead_time_s
                    # Re-check mine-avoidance at the LED landing — leading moves it
                    # off the fish's cell, so 'aimed at a fish = safe' no longer
                    # holds; drop the lead (aim at the fish) on a mine-only spot.
                    if lands_on_mine_only(origin_x + led_dist, mines, fish):
                        led_dist, lead_px_eff = base_dist, 0.0
                    lead_clamped = 1 if clamped else 0
                    target_charge = charge_for_distance(model, led_dist)
                else:
                    target_charge = charge_for_distance(model, base_dist)
                aim_mode = "model_lead" if lead_arm else "model_nolead"
        predicted = distance_for_charge(model, target_charge) if model else None

        # Closed-loop cast: hold while polling the charge bar, release at
        # target_charge. The cast distance is set by the charge level we release
        # at (not where the click lands), so there's no sampled-then-stale
        # latency to fight as in the click-timing games — but no disk writes go
        # between the decision and the hold. The read_charge closure grabs the
        # play crop and reads the left-edge fill each poll.
        fired_at = datetime.now().isoformat(timespec="milliseconds")
        cx = win_left + play["left"] + play["width"] // 2
        cy = win_top + play["top"] + play["height"] // 2

        # The LIVE charge meter is a vertical red thermometer LEFT of the cast
        # bar (find_charge_fill), read from the FULL window anchored to the cast
        # bar in full-window coords (the play crop's x<10 misses it entirely —
        # the thermometer renders left of the crop, #58). The closure tracks the
        # PEAK fill (and dumps each poll's full frame with --save-frames).
        charge_attempt += 1
        charge_dbg = {"peak": 0, "polls": 0, "attempt": charge_attempt}
        cast_bar_full = (play["left"] + bar[0], play["top"] + bar[1], bar[2], bar[3])

        # PTS score BEFORE the cast — the authoritative catch signal (the game's
        # own count). Full window, anchored to the cast bar. Closed-loop timing
        # isn't biased by a pre-hold read (the cast power is the release fill, not
        # a sampled instant), so this sits safely before the hold. None until the
        # digits are captured / on an unreadable frame -> the delta falls back to
        # the bobber-disappearance heuristic below (#58/#63). The crop is saved
        # AFTER the cast (deferred disk write — never between decision and hold).
        full_before = grab_region(win_left, win_top, win_w, win_h)
        score_before = read_score(full_before, cast_bar_full)

        def _read_charge():
            # Fast poll: grab + read only. NO per-poll frame save — writing a
            # full-window PNG each poll stalled the loop ~40ms and the fast-
            # rising bar overshot the target by 20+ on those polls (run 11: the
            # 3 big overshoots were all save-stalls). The bar is located now;
            # find_charge_fill is validated, so the diagnostic save is gone.
            full = grab_region(win_left, win_top, win_w, win_h)
            c = find_charge_fill(full, cast_bar_full)
            charge_dbg["peak"] = max(charge_dbg["peak"], c)
            charge_dbg["polls"] += 1
            return c

        hold_t0 = time.monotonic()
        charge, ready = charge_and_release(
            cx, cy, target_charge, _read_charge,
            poll_s=CHARGE_POLL_S, ready_grace_s=CHARGE_READY_GRACE_S,
            max_hold_s=CHARGE_MAX_HOLD_S)
        # Measured closed-loop hold (the data to fit the lead's decision->landing
        # time and replace LEAD_TIME_FALLBACK_S; hold_ms was logged None before).
        hold_ms = round((time.monotonic() - hold_t0) * 1000)

        if not ready:
            # Rod not ready: the fill never crossed the floor within the grace
            # (previous lure still reeling, OR the bar didn't start charging in
            # time). Skip the cast — don't log/count it — and retry after a beat.
            # The peak/polls report makes the not-charging vs not-ready split
            # visible without needing the saved frames.
            stats["n_not_ready"] += 1
            print(f"rod not ready: peak charge {charge_dbg['peak']} over "
                  f"{charge_dbg['polls']} polls / {CHARGE_READY_GRACE_S}s — waiting")
            time.sleep(NOT_READY_RETRY_S)
            continue
        stats["n_casts"] += 1

        tag = (f"{target['kind']}@{target['target_dist']}px" if target else "explore")
        print(f"cast #{stats['n_casts']} charge={charge} (aim {target_charge}) "
              f"[{aim_mode}] -> {tag} "
              f"(fish={len(fish)} mines={len(mines)} streak={streak})")

        row_id = log_cast(
            db,
            session_started=session_started,
            attempt_idx=1,
            cast_idx=stats["n_casts"],
            fired_at=fired_at,
            hold_ms=hold_ms,        # measured closed-loop hold (for the lead time fit)
            aim_mode=aim_mode,
            cast_origin_x=origin_x,
            cast_origin_y=origin_y,
            target_kind=(target["kind"] if target else "explore"),
            target_x=(target["x"] if target else None),
            target_dist_px=(target["target_dist"] if target else None),
            target_charge_level=target_charge,
            predicted_dist_px=(int(predicted) if predicted is not None else None),
            streak_before=streak,
            n_fish=len(fish),
            n_mines=len(mines),
            window_w=win_w,
            window_h=win_h,
            score_before=score_before,
            arm_fish_vx_px_s=lead_vx,
            lead_time_s=lead_time_s,
            lead_px_intended=(lead_px_int or None),
            lead_px_effective=lead_px_eff,
            lead_clamped=lead_clamped,
            lead_n_samples=(lead_n or None),
            code_commit=code_commit,
            source="bot",
        )

        # Locate the bobber landing (max-x poll) to classify the hit and supply
        # landed_dist (the charge->distance model's training target). The cast
        # power is `charge` from the release above — the robust, in-crop signal.
        lure, post, pre_frame = _measure_landing(win_left, win_top, play,
                                                 CAST_SETTLE_S, LANDING_POLL_S,
                                                 frames_dir, stats["n_casts"])

        # PTS score AFTER the landing — read immediately to minimise the lag
        # between the catch (score ticks when the fish vanishes at landing) and
        # the read. score_after - score_before is the ground-truth catch below.
        full_after = grab_region(win_left, win_top, win_w, win_h)
        score_after = read_score(full_after, cast_bar_full)

        landed_x = landed_y = landed_dist = landed_kind = points = made = None
        catch_dx = catch_dy = None
        detect_source = None
        pre_fish = post_fish = None    # set on a measured landing -> kind attribution (#66)
        if lure is not None:
            landed_x, landed_y = lure
            # Catch = a fish near the landing JUST BEFORE the lure arrives is gone
            # after it lands (a caught fish vanishes); a fish still there = miss.
            pre_fish = find_fish(pre_frame, bar=bar) if pre_frame is not None else []
            post_fish = find_fish(post, bar=bar)
            landed_kind, made = _classify_catch(pre_fish, post_fish, landed_x)
            detect_source = "landing"
            landed_dist = abs(landed_x - origin_x)
            # Catch-geometry telemetry: lure offset to the nearest pre-cast fish
            # (lure - fish). made doesn't track the x gap alone, so log dx AND dy.
            nearest = min(pre_fish, key=lambda f: abs(f["x"] - landed_x), default=None)
            if nearest is not None:
                catch_dx = landed_x - nearest["x"]
                catch_dy = landed_y - nearest["y"]
        elif target is not None and target["kind"] == "eel":
            # Far eel casts often don't settle a measurable bobber, so the
            # landing-based test never runs and the catch goes unrecorded (run
            # 19 caught 2 eels logged as nothing). But an eel (<=1 on the bar,
            # and the cast aimed right at it) is caught if it's GONE from the bar
            # afterward — check that directly (#63).
            after = grab_region(win_left + play["left"], win_top + play["top"],
                                play["width"], play["height"])
            if find_eel(after, bar=find_cast_bar(after)) is None:
                landed_kind, made, landed_x = "eel", 1, target["x"]
                detect_source = "eel_absence"

        # The score delta overrides the bobber/eel heuristic when it's readable
        # (see _resolve_catch) — the game's own count, authoritative.
        landed_kind, made, detect_source = _resolve_catch(
            (landed_kind, made, detect_source), score_before, score_after)

        # Attribute the catch's KIND(s) and per-fish COUNT from the fish that
        # vanished near the landing, cross-checked against the score delta — what
        # the delta alone can't tell (a +2 is a lone eel OR two greens; a +5
        # 'multi' is eel+squid). Refines only the cosmetic landed_kind label and
        # the per-fish streak count (#66/#70); points + made already come from the
        # authoritative delta. Only on a delta-confirmed catch with a MEASURED
        # landing (pre_fish set) — far-cast rescues keep the delta guess + a +1
        # streak (the cross-check can't run with no fish frames).
        n_caught = 1
        if (made and detect_source == "score" and pre_fish is not None
                and landed_x is not None
                and score_before is not None and score_after is not None):
            caught = vanished_fish(pre_fish, post_fish or [], landed_x)
            label, count = attribute_catch(caught, score_after - score_before)
            if label is not None:
                landed_kind = label     # 'eel+squid' etc. (was the opaque 'multi')
                n_caught = count

        if made:
            # Points: the score DELTA is the game's own count, so it's the truth
            # even for a converged multi-catch (landed_kind 'multi'/'eel+squid' has
            # no single FISH_VALUE — its points are the delta). The heuristic paths
            # fall back to the kind's value.
            if detect_source == "score" and score_before is not None and score_after is not None:
                points = score_after - score_before
            else:
                points = FISH_VALUE.get(landed_kind, 0)
            stats["points_total"] += points
            # Streak: advance by the NUMBER OF FISH caught (the game counts per
            # fish, so a converged multi-catch is +2/+3, not +1; n_caught from the
            # vanished-fish attribution, 1 when unmeasured, #70) — except a Whale
            # resets to 1 (wiki). _streak_after handles both, incl. a converged
            # label that contains a whale.
            streak = _streak_after(streak, landed_kind, n_caught)
            stats["max_streak"] = max(stats["max_streak"], streak)
        elif made == 0:
            streak = 0          # a measured miss breaks the streak
        # made is None (the cast wasn't measured) -> streak unchanged
        set_outcome(
            db, row_id,
            landed_x=landed_x,
            landed_y=landed_y,
            landed_dist_px=landed_dist,
            landed_kind=landed_kind,
            points=points,
            made=made,
            charge_level=charge or None,
            catch_dx=catch_dx,
            catch_dy=catch_dy,
            score_after=score_after,
            detect_source=detect_source,
        )

        # Deferred debug saves of the before/after PTS crops (never between the
        # fire decision and the hold). --save-frames feeds fishing-capture-digits
        # to grow the digit-template library (the score climbs 0->9 across casts).
        if frames_dir is not None:
            cb = score_crop(full_before, cast_bar_full)
            if cb is not None:
                save_frame(frames_dir / f"score_c{stats['n_casts']:02d}_before_r{score_before}.png", cb)
            ca = score_crop(full_after, cast_bar_full)
            if ca is not None:
                save_frame(frames_dir / f"score_c{stats['n_casts']:02d}_after_r{score_after}.png", ca)

        random_delay(int(CAST_COOLDOWN_S * 1000), int(CAST_COOLDOWN_S * 1000) + 200)


if __name__ == "__main__":
    run()
