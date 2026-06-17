"""The Launcher shell: the notebook, shared state, and the log poll loop.

The shell owns the cross-tab state (process registry, log queue, status
labels, widget refs) and the single Tk poll loop that drains log_queue;
each tab module builds its own widgets into the shell and reads/writes that
state via the `app` reference passed to its build()/helpers.
"""
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import ImageTk

from ui.launcher import books_tab, bots_tab, cards_tab, frames_tab, process, setup_tab, sql_tab, theme
from ui.launcher.config import MINIGAMES


class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Idleon bot launcher")
        self.root.geometry("980x720")
        self.root.minsize(820, 560)
        theme.apply_theme(self.root)

        self.processes: dict[str, subprocess.Popen | None] = {m["name"]: None for m in MINIGAMES}
        self.status_labels: dict[str, ttk.Label] = {}
        self.setup_buttons: dict[str, tuple[ttk.Button, str]] = {}
        # Running setup/observe tools, keyed by entry point, so the launcher
        # can stop the long-running ones (observe, capture, watch-wind) —
        # they have no other stop control in the GUI.
        self.oneshot_procs: dict[str, subprocess.Popen | None] = {}
        self.log_queue: queue.Queue = queue.Queue()
        # Maps (minigame_name, env_var_name) -> Tk StringVar holding the
        # currently-selected option value for that minigame's bot.
        self.bot_option_vars: dict[tuple[str, str], tk.StringVar] = {}
        # Hoops predictor comparison cards (Pokemon-style, one per kind).
        # Map predictor_kind -> dict of widget refs for fast updating.
        # Built in bots_tab when the hoops row is constructed.
        self.predictor_cards: dict[str, dict] = {}
        # Per-game digit-template status labels (template OCR coverage),
        # keyed by minigame name. Updated on launcher open + after each
        # template-refresh button click.
        self.template_status_labels: dict[str, ttk.Label] = {}
        # Snapshot of the shared hoops cooldown anchored to the save
        # mtime it came from: (save_mtime, cd_at_save_time). The display
        # extrapolates remaining = cd - (now - save_mtime). Anchoring
        # to save_mtime (not read time) keeps the math correct across
        # save flushes and surfaces save-staleness as a separate signal.
        self.hoops_cooldown_snapshot: tuple[float, float] | None = None
        # Estimated daily-reset time-of-day, derived from logged plays-drop
        # boundaries (#22). dict {"hour","minute","samples","spread_min"}
        # or None when there isn't enough data yet. Recomputed on launcher
        # open + on manual Refresh; the per-second tick renders the live
        # countdown from it.
        self.reset_estimate: dict | None = None
        # Last-seen save file mtime, so the periodic display loop can
        # re-pull when Idleon writes a fresh save (e.g. after a hoops
        # session ends in-game and the cooldown jumps to its full value).
        self.last_save_mtime: float = 0.0

        # Frames tab state — keep PhotoImage refs alive so Tk doesn't GC them.
        self.frame_images: list[ImageTk.PhotoImage] = []
        # Maps listbox display name -> Path of the directory it represents.
        self.frame_dirs: dict[str, Path] = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log_queue()

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        bots_frame = ttk.Frame(nb)
        setup_frame = ttk.Frame(nb)
        frames_frame = ttk.Frame(nb)
        sql_frame = ttk.Frame(nb)
        books_frame = ttk.Frame(nb)
        cards_frame = ttk.Frame(nb)
        nb.add(bots_frame, text="Bots")
        nb.add(setup_frame, text="Setup")
        nb.add(frames_frame, text="Frames")
        nb.add(sql_frame, text="SQL")
        nb.add(books_frame, text="Books")
        nb.add(cards_frame, text="Cards")

        bots_tab.build(bots_frame, self)
        setup_tab.build(setup_frame, self)
        frames_tab.build(frames_frame, self)
        sql_tab.build(sql_frame, self)
        books_tab.build(books_frame, self)
        cards_tab.build(cards_frame, self)

    def enqueue_log(self, text: str):
        self.log_queue.put(text)

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item:
                    if item[0] == "status":
                        _, name, text, color = item
                        if name in self.status_labels:
                            self.status_labels[name].config(text=text, foreground=color)
                    elif item[0] == "setup_done":
                        _, entry_point = item
                        self.oneshot_procs[entry_point] = None
                        if entry_point in self.setup_buttons:
                            btn, label = self.setup_buttons[entry_point]
                            # Restore the launch button (it became a Stop
                            # button while the tool ran — see process.spawn).
                            btn.config(
                                state="normal", text=label, style="TButton",
                                command=lambda c=entry_point: process.run_oneshot(self, c),
                            )
                    elif item[0] == "refresh_hoops_stats":
                        bots_tab.refresh_hoops_stats(self)
                    elif item[0] == "refresh_tries":
                        bots_tab.refresh_tries(self, silent=True)
                else:
                    self._append_log(str(item))
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log_queue)

    def _append_log(self, text: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _on_close(self):
        for name, proc in list(self.processes.items()):
            if proc is None:
                continue
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                proc.terminate()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run():
    Launcher().run()
