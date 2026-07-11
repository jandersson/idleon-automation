"""The Launcher shell: header bar, notebook, event bus, poll loop.

Architecture (2026-07-11 refactor):
- The shell owns cross-tab state (process registry, persisted UI state)
  and a single thread-safe EVENT BUS: background threads call app.emit
  (queue put), the Tk poll loop dispatches to handlers registered with
  app.on. This replaces the old magic-tuple queue protocol.
- The header bar (title + tries/cooldown/reset readouts) is part of the
  shell so it stays visible on every tab.
- Each tab module builds its widgets into a notebook page via
  build(parent, app); the Bots tab additionally registers event handlers
  for bot/tool lifecycle events emitted by process.py.
"""
import queue
import subprocess
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from PIL import ImageTk

from common import tries_counter
from common.hoops_cooldown_observer import (
    record_observation as _record_hoops_cooldown_observation,
    estimate_daily_reset,
    next_daily_reset,
)
from common.idleon_save import read_minigame_plays_shared, read_hoops_cooldown
from ui.launcher import (
    books_tab, bots_tab, cards_tab, frames_tab, setup_tab,
    sql_tab, stamps_tab, state, theme,
)
from ui.launcher.config import MINIGAMES

TABS = [
    ("🤖 Bots", bots_tab),
    ("🔧 Setup", setup_tab),
    ("🖼 Frames", frames_tab),
    ("🗃 SQL", sql_tab),
    ("📚 Books", books_tab),
    ("🃏 Cards", cards_tab),
    ("🪙 Stamps", stamps_tab),
]


class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Idleon bot launcher")
        self.root.geometry("1000x760")
        self.root.minsize(860, 580)
        theme.apply_theme(self.root)

        # --- cross-tab state ---
        self.ui_state = state.load()
        self.processes: dict[str, subprocess.Popen | None] = {
            m["name"]: None for m in MINIGAMES}
        self.oneshot_procs: dict[str, subprocess.Popen | None] = {}
        # (minigame_name, env_var) -> StringVar of the selected option.
        self.bot_option_vars: dict[tuple[str, str], tk.StringVar] = {}
        # Registered setup-tool buttons: entry_point -> (Button, label).
        self.setup_buttons: dict[str, tuple[ttk.Button, str]] = {}
        # Hoops predictor cards (built by bots_tab when hoops row exists).
        self.predictor_cards: dict[str, dict] = {}
        self.template_status_labels: dict[str, ttk.Label] = {}
        # Hoops cooldown snapshot anchored to the save mtime it came from:
        # (save_mtime, cd_at_save_time); display extrapolates from it.
        self.hoops_cooldown_snapshot: tuple[float, float] | None = None
        # Estimated daily-reset time-of-day from logged plays-drop
        # boundaries (#22); None until enough data.
        self.reset_estimate: dict | None = None
        self.last_save_mtime: float = 0.0

        # Frames tab state — PhotoImage refs must outlive Tk's GC.
        self.frame_images: list[ImageTk.PhotoImage] = []
        self.frame_dirs: dict[str, Path] = {}

        # --- event bus ---
        self._queue: queue.Queue = queue.Queue()
        self._handlers: dict[str, list] = {}
        self.on("log", lambda text: self._append_log(text))

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_queue()
        self._tick()

    # ---- event bus -------------------------------------------------------
    def emit(self, event: str, *args) -> None:
        """Queue an event from any thread; dispatched on the Tk thread."""
        self._queue.put((event, args))

    def on(self, event: str, handler) -> None:
        """Register a handler (called on the Tk thread with the emit args)."""
        self._handlers.setdefault(event, []).append(handler)

    def _poll_queue(self) -> None:
        try:
            while True:
                event, args = self._queue.get_nowait()
                for handler in self._handlers.get(event, []):
                    try:
                        handler(*args)
                    except Exception as e:  # a bad handler must not kill the loop
                        self._append_log(f"[ui] {event} handler failed: {e!r}\n")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    # Back-compat shim for tabs that log directly.
    def enqueue_log(self, text: str) -> None:
        self.emit("log", text)

    # ---- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        self._build_header()
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=theme.PAD_S, pady=(0, theme.PAD_S))
        for title, module in TABS:
            page = ttk.Frame(nb)
            nb.add(page, text=title)
            module.build(page, self)

    def _build_header(self) -> None:
        """Top bar: wordmark + the account readouts (tries, hoops cooldown,
        daily reset). Lives above the notebook so it's visible on every
        tab — these numbers gate whether playing is even possible."""
        bar = ttk.Frame(self.root, style="Header.TFrame", padding=(theme.PAD_L, theme.PAD))
        bar.pack(fill="x")
        ttk.Label(bar, text="🌳 Idleon Launcher",
                  style="HeaderTitle.TLabel").pack(side="left")

        def chip(caption: str) -> ttk.Label:
            box = ttk.Frame(bar, style="Header.TFrame")
            box.pack(side="right", padx=(theme.PAD_L, 0))
            ttk.Label(box, text=caption, style="HeaderMuted.TLabel").pack(anchor="e")
            value = ttk.Label(box, text="—", style="HeaderValue.TLabel")
            value.pack(anchor="e")
            return value

        # Right-to-left packing: refresh button outermost, then chips.
        ttk.Button(bar, text="⟳", width=3, style="Ghost.TButton",
                   command=self.refresh_account).pack(side="right", padx=(theme.PAD_L, 0))
        self.reset_label = chip("🔄 daily reset")
        self.hoops_label = chip("🏀 cooldown 🚧")
        self.tries_label = chip("🪓🪰⛏ tries")

        self.refresh_account(silent=True)

    # ---- account readouts (save-file derived) ------------------------------
    def refresh_account(self, silent: bool = False) -> None:
        """Re-read tries + hoops cooldown from the local Idleon save and
        recompute the daily-reset estimate."""
        cd = read_hoops_cooldown()
        if cd is not None:
            # Anchor to save mtime, not read time: cd reflects the cooldown
            # at save time, so extrapolation stays exact across flushes.
            anchor = _save_mtime() or time.time()
            self.hoops_cooldown_snapshot = (anchor, cd)
            try:
                # Feeds the cooldown-formula work (#22); gitignored output.
                _record_hoops_cooldown_observation()
            except Exception as e:
                if not silent:
                    self.emit("log", f"[cooldown-obs] skipped: {e}\n")
        plays = read_minigame_plays_shared()
        if plays is None:
            self.tries_label.config(text="?")
            if not silent:
                self.emit("log", "[tries] couldn't read save (Idleon not "
                                 "installed, or plyvel missing)\n")
        else:
            self.tries_label.config(text=str(plays))
            tries_counter.write(plays)  # persist for external consumers
            if not silent:
                self.emit("log", f"[tries] refreshed: {plays}\n")
        try:
            self.reset_estimate = estimate_daily_reset()
        except Exception as e:
            self.reset_estimate = None
            if not silent:
                self.emit("log", f"[reset-est] skipped: {e}\n")

    def _tick(self) -> None:
        """1 Hz UI tick: live hoops-cooldown countdown, reset countdown,
        save-flush detection, and per-bot runtime timers (via event)."""
        mtime = _save_mtime()
        if mtime > self.last_save_mtime:
            self.last_save_mtime = mtime
            self.refresh_account(silent=True)
        self._render_hoops_label()
        self._render_reset_label()
        self.emit("tick")
        self.root.after(1000, self._tick)

    def _render_hoops_label(self) -> None:
        snap = self.hoops_cooldown_snapshot
        if snap is None:
            self.hoops_label.config(text="—")
            return
        save_at, stored_cd = snap
        now = time.time()
        remaining = stored_cd - (now - save_at)
        save_age = now - save_at
        stale = save_age > 30
        if remaining <= 0:
            # Don't claim "ready" off a stale snapshot — the user may have
            # played since the last flush (verified misleading 2026-05-10).
            text = "?" if stale else "ready"
        else:
            m, s = divmod(int(remaining) + 1, 60)
            text = f"{m}:{s:02d}"
        if stale:
            am, asec = divmod(int(save_age), 60)
            text += f" ({am}m old)" if am else f" ({asec}s old)"
        self.hoops_label.config(text=text)

    def _render_reset_label(self) -> None:
        est = self.reset_estimate
        nxt = next_daily_reset(est)
        if nxt is None:
            self.reset_label.config(text="?")
            return
        rem = max(0, int((nxt - datetime.now()).total_seconds()))
        h, r = divmod(rem, 3600)
        countdown = f"{h}h{r // 60:02d}m" if h else f"{r // 60}m"
        rough = "?" if est.get("samples", 0) < 2 or est.get("spread_min", 0.0) > 20.0 else ""
        self.reset_label.config(text=f"~{est['hour']:02d}:{est['minute']:02d}{rough} · {countdown}")

    # ---- log pane (widget owned by bots_tab; writer lives here) ------------
    def _append_log(self, text: str) -> None:
        widget = getattr(self, "log_text", None)
        if widget is None:
            return
        widget.config(state="normal")
        # Per-source coloring: lines look like "[source] ...".
        tag = None
        if text.startswith("["):
            source = text[1:text.find("]")] if "]" in text else ""
            tag = bots_tab.log_tag_for(self, source)
        widget.insert("end", text, tag)
        if getattr(self, "log_autoscroll", None) is None or self.log_autoscroll.get():
            widget.see("end")
        widget.config(state="disabled")

    # ---- lifecycle ---------------------------------------------------------
    def _on_close(self) -> None:
        for proc in list(self.processes.values()) + list(self.oneshot_procs.values()):
            if proc is None:
                continue
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                proc.terminate()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _save_mtime() -> float:
    """Newest mtime among the Idleon LevelDB save files, 0 if unreachable."""
    import glob
    import os
    save_dir = os.path.expandvars(r"%APPDATA%\legends-of-idleon\Local Storage\leveldb")
    try:
        files = (glob.glob(os.path.join(save_dir, "*.log"))
                 + glob.glob(os.path.join(save_dir, "*.ldb")))
        return max((os.path.getmtime(p) for p in files), default=0.0)
    except OSError:
        return 0.0


def run():
    Launcher().run()
