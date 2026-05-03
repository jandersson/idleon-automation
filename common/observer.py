"""Generic 'watch me play' framework for minigame observation.

Listens for the user's left-clicks inside the named game window, calls a
per-minigame callback with the captured frame, and shows a small
click-through Tk overlay indicating cooldown state. The cooldown is
mostly there so the user knows when score/state changes have settled
enough to be reliably read.

Each minigame supplies an `on_click(idx, click_x_rel, click_y_rel, pre_frame)`
callback. The framework handles:
  - pynput mouse listener
  - Filtering clicks to inside the game window
  - Per-click frame capture (passed to the callback)
  - Cooldown indicator overlay (red countdown vs green GO)
  - Fail-safe abort (slam mouse to a screen corner)

The callback owns whatever's game-specific: running detectors, scheduling
post-shot work via threading.Timer, logging to a DB or to disk, etc.

For a brand-new minigame with no detectors yet, the simplest useful
callback is to just save the captured frame to disk so you can browse
the visual evidence afterwards (see `simple_frame_saver`).
"""
import ctypes
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pyautogui
import pynput

from common.capture import grab_region
from common.input import check_failsafe
from common.window import get_bounds, WindowNotFoundError


@dataclass
class ObserverConfig:
    window_title: str
    cooldown_seconds: float = 4.0
    overlay_size: int = 80
    # Optional: a function the overlay calls every `stop_check_interval`
    # seconds. If it returns True, the overlay closes and the observer
    # exits cleanly. Use this for game-over detection so the user
    # doesn't have to fail-safe out manually after the round ends.
    stop_check: "Callable[[], bool] | None" = None
    stop_check_interval: float = 2.0


# (idx, click_x_rel, click_y_rel, pre_frame_bgra) -> None
ClickCallback = Callable[[int, int, int, np.ndarray], None]


def run_observer(config: ObserverConfig, on_click: ClickCallback) -> int:
    """Run the observer until the user fail-safes out. Returns total
    captured-click count."""
    shot_idx = [0]
    cooldown_until = [0.0]

    def click_handler(screen_x: int, screen_y: int, button, pressed: bool) -> None:
        if not pressed or button != pynput.mouse.Button.left:
            return
        try:
            left, top, width, height = get_bounds(config.window_title)
        except WindowNotFoundError:
            return
        if not (left <= screen_x <= left + width and top <= screen_y <= top + height):
            return
        click_x_rel = int(screen_x - left)
        click_y_rel = int(screen_y - top)
        frame = grab_region(left, top, width, height)
        shot_idx[0] += 1
        cooldown_until[0] = time.time() + config.cooldown_seconds
        try:
            on_click(shot_idx[0], click_x_rel, click_y_rel, frame)
        except Exception as e:
            print(f"  [observer] callback error: {e}")

    listener = pynput.mouse.Listener(on_click=click_handler)
    listener.start()
    try:
        _run_overlay(config, cooldown_until)
    except pyautogui.FailSafeException:
        print("Fail-safe triggered — exiting observe mode.")
    except KeyboardInterrupt:
        print("Interrupted — exiting observe mode.")
    finally:
        listener.stop()
        listener.join(timeout=1)
    return shot_idx[0]


def simple_frame_saver(out_dir: Path) -> ClickCallback:
    """A minimal `on_click` that just saves the pre-shot frame to disk.

    Useful as a starting point for a new minigame where you don't yet
    know what to detect — save everything, look at it offline."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_click(idx: int, x: int, y: int, frame: np.ndarray) -> None:
        ts = datetime.now().strftime("%H%M%S")
        path = out_dir / f"click_{idx:03d}_{ts}_x{x}_y{y}.png"
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(path), bgr)
        print(f"  click {idx}: saved {path.name}")

    return on_click


def _run_overlay(config: ObserverConfig, cooldown_until: list) -> None:
    """Tk overlay window: red countdown during cooldown, green GO when
    ready. Click-through on Windows so user clicks reach the game."""
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.85)
    size = config.overlay_size
    canvas = tk.Canvas(root, width=size, height=size,
                       highlightthickness=0, bg="black")
    canvas.pack()
    root.update_idletasks()

    if sys.platform == "win32":
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT,
        )

    last_stop_check = [0.0]

    def tick() -> None:
        try:
            check_failsafe()
        except pyautogui.FailSafeException:
            root.destroy()
            return
        # Periodic stop check (e.g. for game-over detection).
        if config.stop_check is not None:
            now = time.time()
            if now - last_stop_check[0] >= config.stop_check_interval:
                last_stop_check[0] = now
                try:
                    if config.stop_check():
                        print("  [observer] stop_check returned True — exiting.")
                        root.destroy()
                        return
                except Exception as e:
                    print(f"  [observer] stop_check error: {e}")
        try:
            left, top, w, h = get_bounds(config.window_title)
            x = left + w - size - 10
            y = top + 10
            root.geometry(f"{size}x{size}+{x}+{y}")
        except WindowNotFoundError:
            pass
        now = time.time()
        remaining = cooldown_until[0] - now
        canvas.delete("all")
        if remaining > 0:
            canvas.create_oval(8, 8, size - 8, size - 8,
                               fill="#c33", outline="")
            canvas.create_text(size // 2, size // 2,
                               text=f"{remaining:.1f}",
                               fill="white", font=("Segoe UI", 16, "bold"))
        else:
            canvas.create_oval(8, 8, size - 8, size - 8,
                               fill="#2a2", outline="")
            canvas.create_text(size // 2, size // 2,
                               text="GO", fill="white",
                               font=("Segoe UI", 16, "bold"))
        root.after(80, tick)

    tick()
    root.mainloop()
