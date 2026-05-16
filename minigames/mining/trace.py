"""Continuous-capture + click-log tracer for the mining minigame.

Purpose: answer "does the cart auto-scroll after pressing Start, or only
after the first jump click?" (issue #1). You start this once from a
terminal, alt-tab to the game during the startup window, then play one
attempt naturally — clicks are logged with frame indices so we can
correlate motion-in-frames with input events.

Outputs to assets/captures/trace_<stamp>/:
  - frame_NNN.png — one image per tick
  - trace.json — {frames: [...], clicks: [{frame_idx, t_rel, x, y}], ...}

Failsafe: slam mouse to any screen corner to abort early.
"""
import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import pyautogui
import pynput

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import check_failsafe, click as bot_click
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from minigames.mining.main import WINDOW_TITLE

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"
CAPTURES_DIR = _HERE / "assets" / "captures"


def run():
    parser = argparse.ArgumentParser(description="Trace mining: continuous frames + click log")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="trace duration after startup (default 30)")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="seconds between frames (default 0.1)")
    parser.add_argument("--startup", type=float, default=8.0,
                        help="seconds before capture starts (default 8) — use this to alt-tab to game")
    parser.add_argument("--no-bot-click", action="store_true",
                        help="don't click Play Game — user plays manually, bot only records frames + clicks")
    args = parser.parse_args()

    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner(args)


def _run_inner(args):
    try:
        left, top, width, height = get_bounds(WINDOW_TITLE)
    except WindowNotFoundError as e:
        print(e)
        return

    if args.no_bot_click:
        btn_cx_rel = btn_cy_rel = None
        print(f"Window {WINDOW_TITLE!r} at ({left},{top}) size {width}x{height}")
        print(f"Startup: {args.startup:.0f}s — alt-tab to the game.")
        print(f"Manual mode: you play, bot only records {args.seconds:.0f}s of frames @ "
              f"{args.interval*1000:.0f}ms + click timestamps.")
        print("Slam mouse to any screen corner to abort.\n")
    else:
        start_btn = get_region(_HERE, "start_button", width, height)
        if start_btn is None:
            print("No 'start_button' region in regions.json. Run mining-pick-start-button first.")
            return
        btn_cx_rel = start_btn["left"] + start_btn["width"] // 2
        btn_cy_rel = start_btn["top"] + start_btn["height"] // 2
        print(f"Window {WINDOW_TITLE!r} at ({left},{top}) size {width}x{height}")
        print(f"Start button center (window-rel): ({btn_cx_rel}, {btn_cy_rel})")
        print(f"Startup: {args.startup:.0f}s — bring the minigame Start prompt on-screen.")
        print(f"Bot will click Start, then capture for {args.seconds:.0f}s @ {args.interval*1000:.0f}ms intervals.")
        print("Slam mouse to any screen corner to abort.\n")

    for s in range(int(args.startup), 0, -1):
        print(f"  starting in {s}...")
        time.sleep(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = CAPTURES_DIR / f"trace_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing frames to {out_dir}")

    click_log: list[dict] = []
    click_lock = threading.Lock()

    def on_click(screen_x: int, screen_y: int, button, pressed: bool) -> None:
        if not pressed or button != pynput.mouse.Button.left:
            return
        try:
            wl, wt, ww, wh = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError:
            return
        if not (wl <= screen_x <= wl + ww and wt <= screen_y <= wt + wh):
            return
        with click_lock:
            click_log.append({
                "t_rel": round(time.time() - t0, 3),
                "x_rel": int(screen_x - wl),
                "y_rel": int(screen_y - wt),
                "source": "human",
            })

    if btn_cx_rel is None:
        # Manual mode — T=0 is just the moment after startup countdown.
        t0 = time.time()
    else:
        # Fire the Start click. T=0 is the moment immediately after this
        # click returns.
        btn_screen_x = left + btn_cx_rel
        btn_screen_y = top + btn_cy_rel
        print(f"Clicking Start at screen ({btn_screen_x}, {btn_screen_y})")
        bot_click(btn_screen_x, btn_screen_y)
        t0 = time.time()
        click_log.append({
            "t_rel": 0.0,
            "x_rel": btn_cx_rel,
            "y_rel": btn_cy_rel,
            "source": "bot:start",
        })

    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()

    aborted = False
    n_frames = int(args.seconds / args.interval)
    frame_times: list[float] = []
    try:
        for i in range(n_frames):
            try:
                check_failsafe()
            except pyautogui.FailSafeException:
                print("Failsafe triggered — stopping early.")
                aborted = True
                break
            try:
                wl, wt, ww, wh = get_bounds(WINDOW_TITLE)
            except WindowNotFoundError:
                print("Game window lost — stopping.")
                aborted = True
                break
            frame = grab_region(wl, wt, ww, wh)
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), bgr)
            frame_times.append(round(time.time() - t0, 3))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrupted — saving what we have.")
        aborted = True
    finally:
        listener.stop()
        listener.join(timeout=1)
        _write_metadata(out_dir, args, width, height, frame_times,
                        list(click_log), aborted)
        print(f"\nWrote {len(frame_times)} frames to {out_dir}")


def _write_metadata(out_dir, args, win_w, win_h, frame_times, clicks, aborted):
    """Attach frame-idx to each click and write trace.json. Robust to
    partial state — runs from `finally` so a ctrl-C or unexpected exit
    still produces a usable trace."""
    for c in clicks:
        t = c["t_rel"]
        idx = -1
        for fi, ft in enumerate(frame_times):
            if ft <= t:
                idx = fi
            else:
                break
        c["frame_idx"] = idx
    meta = {
        "window_title": WINDOW_TITLE,
        "window_size": [win_w, win_h],
        "interval_s": args.interval,
        "seconds_requested": args.seconds,
        "n_frames": len(frame_times),
        "aborted": aborted,
        "manual_mode": args.no_bot_click,
        "frame_times": frame_times,
        "clicks": clicks,
    }
    try:
        (out_dir / "trace.json").write_text(json.dumps(meta, indent=2))
    except Exception as e:
        print(f"  [trace] failed to write trace.json: {e}")


if __name__ == "__main__":
    run()
