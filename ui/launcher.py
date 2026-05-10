"""Tkinter launcher for the Idleon bots.

Two tabs:
- Bots: per-minigame Start/Stop and inline setup-tool buttons. Each button
  shells out to `uv run <entry-point>` in a subprocess; stdout streams into
  the log pane at the bottom.
- Frames: image inspector. Pick a minigame, then any directory under its
  assets/ that contains PNGs — see them stacked. Works for hoops/darts
  per-shot monitor folders, capture dumps, calibration outputs, etc.
"""
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from common import tries_counter
from common.idleon_save import read_minigame_plays, read_hoops_cooldown
from common import hoops_cooldown_state

PROJECT_ROOT = Path(__file__).parent.parent
MINIGAMES_DIR = PROJECT_ROOT / "minigames"

MINIGAMES = [
    {
        "name": "chopping",
        "emoji": "🪓",
        "bot": "chopping",
        "setup": [
            ("Observe (you play)", "chopping-observe"),
            ("Pick bar", "chopping-pick-bar-region"),
            ("Pick leaf", "chopping-pick-leaf-region"),
            ("Pick button", "chopping-pick-button-region"),
            ("Calibrate", "chopping-calibrate"),
        ],
    },
    {
        "name": "hoops",
        "emoji": "🏀",
        "bot": "hoops",
        "setup": [
            ("Observe (you play)", "hoops-observe"),
            ("Capture", "hoops-capture"),
            ("Debug match", "hoops-debug"),
            ("Ball calibrate", "hoops-ball-calibrate"),
            ("Score calibrate", "hoops-score-calibrate"),
            ("Pick score", "hoops-pick-score-region"),
            ("Pick game over", "hoops-pick-game-over"),
            ("Pick lives", "hoops-pick-lives-region"),
        ],
        "bot_options": [
            {
                "label": "Predictor",
                "env": "HOOPS_PREDICTOR_KIND",
                "values": ["gp", "knn", "bivariate", "trajectory_knn"],
                "default": "gp",
            },
        ],
    },
    {
        "name": "darts",
        "emoji": "🎯",
        "bot": "darts",
        "setup": [
            ("Observe (you play)", "darts-observe"),
            ("Capture", "darts-capture"),
            ("Pick release", "darts-pick-release"),
            ("Auto-crop release", "darts-auto-crop-release"),
            ("Pick wind", "darts-pick-wind-region"),
            ("Watch wind", "darts-watch-wind"),
            ("Pick score", "darts-pick-score-region"),
        ],
    },
    {
        "name": "catching",
        "emoji": "🪰",
        "bot": "catching",
        "setup": [
            ("Observe (you play)", "catching-observe"),
            ("Capture", "catching-capture"),
            ("Pick play region", "catching-pick-play-region"),
            ("Extract fly", "catching-extract-fly"),
        ],
    },
    {
        "name": "mining",
        "emoji": "⛏️",
        "bot": "mining",
        "setup": [
            ("Observe (you play)", "mining-observe"),
            ("Capture", "mining-capture"),
            ("Pick play region", "mining-pick-play-region"),
        ],
    },
]


class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Idleon bot launcher")
        self.root.geometry("980x720")

        self.processes: dict[str, subprocess.Popen | None] = {m["name"]: None for m in MINIGAMES}
        self.status_labels: dict[str, ttk.Label] = {}
        self.setup_buttons: dict[str, tuple[ttk.Button, str]] = {}
        self.log_queue: queue.Queue = queue.Queue()
        # Maps (minigame_name, env_var_name) -> Tk StringVar holding the
        # currently-selected option value for that minigame's bot.
        self.bot_option_vars: dict[tuple[str, str], tk.StringVar] = {}
        # Hoops predictor comparison label (per-predictor make rate); populated
        # at launcher start and refreshed when a hoops session ends.
        self.hoops_stats_label: ttk.Label | None = None
        # Snapshot of the shared hoops cooldown anchored to the save
        # mtime it came from: (save_mtime, cd_at_save_time). The display
        # extrapolates remaining = cd - (now - save_mtime). Anchoring
        # to save_mtime (not read time) keeps the math correct across
        # save flushes and surfaces save-staleness as a separate signal.
        self.hoops_cooldown_snapshot: tuple[float, float] | None = None
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

        bots_tab = ttk.Frame(nb)
        setup_tab = ttk.Frame(nb)
        frames_tab = ttk.Frame(nb)
        nb.add(bots_tab, text="Bots")
        nb.add(setup_tab, text="Setup")
        nb.add(frames_tab, text="Frames")

        self._build_bots_tab(bots_tab)
        self._build_setup_tab(setup_tab)
        self._build_frames_tab(frames_tab)

    def _build_bots_tab(self, parent: ttk.Frame):
        # Top strip: shared "global tries" counter (chopping + catching share
        # one tries pool in-game; user updates manually since it isn't easily
        # OCR-able — only visible on a pre-game popup).
        self._build_tries_strip(parent).grid(
            row=0, column=0, sticky="ew", padx=8, pady=(8, 0)
        )

        for i, mg in enumerate(MINIGAMES):
            label = f"{mg.get('emoji', '')} {mg['name'].capitalize()}".strip()
            frame = ttk.LabelFrame(parent, text=label, padding=6)
            frame.grid(row=i + 1, column=0, sticky="ew", padx=8, pady=4)

            top = ttk.Frame(frame)
            top.pack(fill="x")
            ttk.Button(top, text="Start", width=8,
                       command=lambda m=mg: self._start_bot(m)).pack(side="left")
            ttk.Button(top, text="Stop", width=8,
                       command=lambda m=mg: self._stop_bot(m)).pack(side="left", padx=(4, 12))
            status = ttk.Label(top, text="stopped", foreground="grey")
            status.pack(side="left")
            self.status_labels[mg["name"]] = status

            for opt in mg.get("bot_options", []):
                var = tk.StringVar(value=opt["default"])
                self.bot_option_vars[(mg["name"], opt["env"])] = var
                ttk.Label(top, text=opt["label"] + ":").pack(side="left", padx=(12, 4))
                ttk.Combobox(
                    top, textvariable=var, state="readonly",
                    values=opt["values"], width=max(len(v) for v in opt["values"]) + 2,
                ).pack(side="left")

            # Hoops gets a per-predictor stats line under the controls so we
            # can A/B compare gp vs trajectory_knn at a glance — see GitHub
            # issue #21. Other minigames don't have a comparable signal yet.
            if mg["name"] == "hoops":
                self.hoops_stats_label = ttk.Label(
                    frame, text="", foreground="grey",
                    font=("Segoe UI", 9),
                )
                self.hoops_stats_label.pack(fill="x", pady=(4, 0))
                self._refresh_hoops_stats()

        log_frame = ttk.LabelFrame(parent, text="Log", padding=4)
        log_frame.grid(row=len(MINIGAMES) + 1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.log_text = tk.Text(log_frame, height=10, wrap="none", state="disabled",
                                bg="#111", fg="#ddd", insertbackground="#ddd",
                                font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        parent.columnconfigure(0, weight=1)
        # Log row is len(MINIGAMES) + 1 (tries strip is row 0).
        parent.rowconfigure(len(MINIGAMES) + 1, weight=1)

    def _build_tries_strip(self, parent: ttk.Frame) -> ttk.Frame:
        """Strip showing the per-character minigame plays remaining
        (chopping + catching + mining share this counter in-game) plus
        the next-playable hoops countdown. Auto-reads from the local
        Idleon save on launcher open; Refresh re-reads on demand."""
        strip = ttk.Frame(parent)
        ttk.Label(strip, text="🪓 / 🪰 / ⛏️ Tries:").pack(side="left")

        self.tries_label = ttk.Label(
            strip, text="—", font=("Segoe UI", 10, "bold"),
        )
        self.tries_label.pack(side="left", padx=(6, 12))

        ttk.Button(strip, text="Refresh", width=8,
                   command=self._refresh_tries).pack(side="left")

        ttk.Label(strip, text="🏀 Cooldown:").pack(side="left", padx=(16, 4))
        self.hoops_label = ttk.Label(
            strip, text="—", font=("Segoe UI", 10, "bold"),
        )
        self.hoops_label.pack(side="left")

        # Read once at startup so values are visible without a manual click.
        self._refresh_tries(silent=True)
        # Tick the hoops countdown every second so it stays live between
        # save flushes.
        self._tick_hoops_display()
        return strip

    def _refresh_hoops_stats(self) -> None:
        """Update the predictor-comparison label from shots.db. Cheap query
        (one GROUP BY over a 400-row table); called on launcher start and
        whenever a hoops session ends."""
        if self.hoops_stats_label is None:
            return
        import sqlite3
        db_path = MINIGAMES_DIR / "hoops" / "assets" / "shots.db"
        if not db_path.exists():
            self.hoops_stats_label.config(text="(no shots.db yet)")
            return
        try:
            con = sqlite3.connect(str(db_path))
            rows = con.execute(
                "SELECT predictor_kind, COUNT(*), SUM(made) "
                "FROM shots WHERE direction='up' AND predictor_kind IS NOT NULL "
                "GROUP BY predictor_kind ORDER BY predictor_kind"
            ).fetchall()
            con.close()
        except sqlite3.Error as e:
            self.hoops_stats_label.config(text=f"(stats error: {e})")
            return
        if not rows:
            self.hoops_stats_label.config(text="(no labelled shots yet)")
            return
        parts = []
        for kind, shots, makes in rows:
            makes = makes or 0
            rate = makes / shots if shots else 0
            parts.append(f"{kind} {makes}/{shots} ({rate:.0%})")
        self.hoops_stats_label.config(text="  ".join(parts))

    def _save_mtime(self) -> float:
        """Newest mtime among the LevelDB save files, or 0 if unreachable."""
        import os, glob
        save_dir = os.path.expandvars(r"%APPDATA%\legends-of-idleon\Local Storage\leveldb")
        try:
            files = glob.glob(os.path.join(save_dir, "*.log")) + glob.glob(os.path.join(save_dir, "*.ldb"))
            return max((os.path.getmtime(p) for p in files), default=0.0)
        except OSError:
            return 0.0

    def _tick_hoops_display(self) -> None:
        """Update the hoops cooldown label, extrapolating from whichever
        source has the most recent cooldown set: the save snapshot or
        the client-side state file written when the bot finishes a
        hoops session. Re-arms itself once a second.

        Picking the source with the higher remaining is equivalent to
        picking the one whose cooldown was set most recently — so a
        bot-finished play immediately reflects in the display, even
        though Idleon hasn't flushed that play to disk yet. The save
        wins back over once Idleon flushes (e.g. on character switch),
        which is correct since the save is then both more recent and
        ground-truth.
        """
        import time
        # Save-mtime watcher: re-pull when Idleon flushes a new save.
        mtime = self._save_mtime()
        if mtime > self.last_save_mtime:
            self.last_save_mtime = mtime
            self._refresh_tries(silent=True)
        now = time.time()
        candidates: list[tuple[str, float, float]] = []  # (source, remaining, age)
        snap = self.hoops_cooldown_snapshot
        if snap is not None:
            save_at, stored_cd = snap
            candidates.append(("save", stored_cd - (now - save_at), now - save_at))
        client = hoops_cooldown_state.read()
        if client is not None:
            elapsed = now - float(client["set_at"])
            candidates.append(
                ("client", float(client["duration_seconds"]) - elapsed, elapsed)
            )
        if not candidates:
            self.hoops_label.config(text="—")
        else:
            # Most-recently-set cooldown wins (= largest remaining).
            source, remaining, age = max(candidates, key=lambda t: t[1])
            if remaining <= 0:
                base = "ready"
            else:
                m, s = divmod(int(remaining) + 1, 60)
                base = f"{m}:{s:02d}"
            # Only flag staleness when the save is the chosen source AND
            # it's old. Client-side data is always fresh by construction.
            if source == "save" and age > 30:
                am, asec = divmod(int(age), 60)
                age_str = f"{am}m{asec:02d}s" if am else f"{asec}s"
                base += f"  (save {age_str} old)"
            self.hoops_label.config(text=base)
        self.root.after(1000, self._tick_hoops_display)

    def _refresh_tries(self, silent: bool = False) -> None:
        import time
        cd = read_hoops_cooldown()
        if cd is not None:
            # Anchor the snapshot to the save mtime, not the read time:
            # the cd value reflects the cooldown *at save time*, so
            # extrapolating from save_mtime keeps the math exact.
            anchor = self._save_mtime() or time.time()
            self.hoops_cooldown_snapshot = (anchor, cd)
        plays = read_minigame_plays()
        if plays is None:
            self.tries_label.config(text="(save not found)")
            if not silent:
                self._enqueue_log("[tries] couldn't read save (Idleon not installed, or plyvel missing)\n")
            return
        if not plays:
            self.tries_label.config(text="0 (no chars)")
            if not silent:
                self._enqueue_log("[tries] save loaded but no characters with MinigamePlays found\n")
            return
        total = sum(plays.values())
        # Per-char breakdown after total.
        per_char = ", ".join(f"{n}={c}" for n, c in plays.items())
        self.tries_label.config(text=f"{total}  ({per_char})")
        # Persist for any external consumer.
        tries_counter.write(total)
        if not silent:
            self._enqueue_log(f"[tries] refreshed: {total} total ({per_char})\n")

    def _build_setup_tab(self, parent: ttk.Frame):
        """Per-minigame calibration / debug buttons. Lives in its own tab so
        the Bots tab stays focused on starting/stopping; setup tools are used
        rarely (mostly during initial calibration)."""
        for i, mg in enumerate(MINIGAMES):
            label = f"{mg.get('emoji', '')} {mg['name'].capitalize()}".strip()
            frame = ttk.LabelFrame(parent, text=label, padding=6)
            frame.grid(row=i, column=0, sticky="ew", padx=8, pady=4)
            for label, cmd in mg["setup"]:
                btn = ttk.Button(frame, text=label,
                                 command=lambda c=cmd: self._run_oneshot(c))
                btn.pack(side="left", padx=2)
                self.setup_buttons[cmd] = (btn, label)
        parent.columnconfigure(0, weight=1)

    def _build_frames_tab(self, parent: ttk.Frame):
        controls = ttk.Frame(parent, padding=6)
        controls.pack(fill="x")

        ttk.Label(controls, text="Minigame:").pack(side="left")
        self.frame_minigame = tk.StringVar(value=MINIGAMES[0]["name"])
        mg_picker = ttk.Combobox(
            controls, textvariable=self.frame_minigame, state="readonly",
            values=[mg["name"] for mg in MINIGAMES],
            width=12,
        )
        mg_picker.pack(side="left", padx=(4, 12))
        mg_picker.bind("<<ComboboxSelected>>", lambda _e: self._refresh_frames_list())

        ttk.Button(controls, text="Refresh", command=self._refresh_frames_list).pack(side="left")
        self.frame_status_label = ttk.Label(controls, text="", foreground="grey")
        self.frame_status_label.pack(side="left", padx=12)

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Left: list of directories under assets/ that contain PNGs.
        list_frame = ttk.LabelFrame(body, text="Folders with images", padding=4)
        list_frame.pack(side="left", fill="y", padx=(0, 6))
        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side="right", fill="y")
        self.frame_listbox = tk.Listbox(list_frame, width=32, height=20,
                                        yscrollcommand=list_scroll.set,
                                        font=("Consolas", 9))
        self.frame_listbox.pack(side="left", fill="y")
        list_scroll.config(command=self.frame_listbox.yview)
        self.frame_listbox.bind("<<ListboxSelect>>", lambda _e: self._show_selected_frames())

        # Right: scrollable image canvas.
        viewer_frame = ttk.LabelFrame(body, text="Images", padding=4)
        viewer_frame.pack(side="left", fill="both", expand=True)

        self.frame_canvas = tk.Canvas(viewer_frame, bg="#222", highlightthickness=0)
        self.frame_canvas.pack(side="left", fill="both", expand=True)
        canvas_scroll = ttk.Scrollbar(viewer_frame, orient="vertical",
                                      command=self.frame_canvas.yview)
        canvas_scroll.pack(side="right", fill="y")
        self.frame_canvas.config(yscrollcommand=canvas_scroll.set)

        self.frame_inner = ttk.Frame(self.frame_canvas)
        self.frame_canvas.create_window((0, 0), window=self.frame_inner, anchor="nw")
        self.frame_inner.bind(
            "<Configure>",
            lambda _e: self.frame_canvas.configure(scrollregion=self.frame_canvas.bbox("all")),
        )
        self.frame_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.frame_canvas.yview_scroll(-int(e.delta / 120), "units"),
        )

        self._refresh_frames_list()

    def _refresh_frames_list(self):
        self.frame_listbox.delete(0, "end")
        self.frame_dirs.clear()
        mg = self.frame_minigame.get()
        assets = MINIGAMES_DIR / mg / "assets"
        if not assets.exists():
            self.frame_status_label.config(text=f"no assets/ dir for {mg}")
            return

        # Find every directory under assets/ (including assets/ itself) that
        # has at least one PNG directly inside it. Display path is relative
        # to assets/ for compactness; "." represents assets/ root.
        candidates: list[Path] = []
        for d in [assets, *(p for p in assets.rglob("*") if p.is_dir())]:
            if any(p.suffix.lower() == ".png" for p in d.iterdir() if p.is_file()):
                candidates.append(d)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for d in candidates:
            rel = "." if d == assets else str(d.relative_to(assets)).replace("\\", "/")
            self.frame_listbox.insert("end", rel)
            self.frame_dirs[rel] = d
        self.frame_status_label.config(
            text=f"{len(candidates)} folder(s) with PNGs under {mg}/assets/"
        )

    def _show_selected_frames(self):
        sel = self.frame_listbox.curselection()
        if not sel:
            return
        rel = self.frame_listbox.get(sel[0])
        folder = self.frame_dirs.get(rel)
        if folder is None or not folder.exists():
            return

        for child in self.frame_inner.winfo_children():
            child.destroy()
        self.frame_images.clear()

        # Show any sibling text files (meta.txt, notes) above the images.
        for txt in sorted(folder.glob("*.txt")):
            try:
                content = txt.read_text(errors="replace")
            except Exception:
                continue
            t = tk.Text(self.frame_inner, height=min(8, max(2, content.count("\n") + 1)),
                        wrap="word", bg="#1a1a1a", fg="#ddd", font=("Consolas", 9))
            t.insert("1.0", f"{txt.name}\n{content}")
            t.config(state="disabled")
            t.pack(fill="x", padx=4, pady=(0, 6))

        # Stable ordering: a few well-known names first (pre/post for hoops),
        # then everything else sorted by name.
        png_files = sorted(folder.glob("*.png"))
        priority = ["pre_shot.png", "post_shot.png"]
        ordered = (
            [p for name in priority for p in png_files if p.name == name]
            + [p for p in png_files if p.name not in priority]
        )
        for path in ordered:
            try:
                img = Image.open(path)
            except Exception as e:
                ttk.Label(self.frame_inner, text=f"{path.name}: {e}",
                          foreground="red").pack(anchor="w")
                continue
            max_w = 720
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.frame_images.append(photo)
            row = ttk.Frame(self.frame_inner)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=path.name, width=20,
                      foreground="#9cf").pack(side="left", anchor="n")
            tk.Label(row, image=photo, bd=0).pack(side="left")

    def _start_bot(self, mg: dict):
        name = mg["name"]
        if self.processes.get(name) is not None:
            self._enqueue_log(f"[{name}] already running\n")
            return
        extra_env: dict[str, str] = {}
        for opt in mg.get("bot_options", []):
            var = self.bot_option_vars.get((name, opt["env"]))
            if var is not None:
                extra_env[opt["env"]] = var.get()
        if extra_env:
            kv = ", ".join(f"{k}={v}" for k, v in extra_env.items())
            self._enqueue_log(f"[{name}] options: {kv}\n")
        self._spawn(mg["bot"], track_as=name, extra_env=extra_env)
        self.status_labels[name].config(text="running", foreground="#3a3")

    def _stop_bot(self, mg: dict):
        name = mg["name"]
        proc = self.processes.get(name)
        if proc is None:
            return
        self._enqueue_log(f"[{name}] stopping...\n")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.terminate()

    def _run_oneshot(self, cmd: str):
        self._spawn(cmd, track_as=None)

    def _spawn(self, entry_point: str, track_as: str | None,
               extra_env: dict[str, str] | None = None):
        import os
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        # PYTHONUNBUFFERED=1 so the bot's stdout flushes line-by-line
        # rather than block-buffering — keeps the launcher log live
        # instead of dumping everything at session end.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if extra_env:
            env.update(extra_env)
        try:
            # --no-sync: skip uv's implicit dependency-sync check. The launcher
            # itself is one of the entry points (idleon.exe), so a sync would
            # try to rewrite a file that's currently locked by us, causing
            # "Access is denied" on Windows. User runs `uv sync` manually
            # when they actually change dependencies.
            proc = subprocess.Popen(
                ["uv", "run", "--no-sync", entry_point],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                creationflags=creationflags,
                env=env,
            )
        except FileNotFoundError:
            self._enqueue_log(f"[{entry_point}] could not run — is `uv` on your PATH?\n")
            return
        if track_as is not None:
            self.processes[track_as] = proc
        if entry_point in self.setup_buttons:
            btn, label = self.setup_buttons[entry_point]
            btn.config(state="disabled", text=f"{label} (running)")
        self._enqueue_log(f"[{entry_point}] started (pid {proc.pid})\n")
        threading.Thread(target=self._drain, args=(entry_point, proc, track_as),
                         daemon=True).start()

    def _drain(self, entry_point: str, proc: subprocess.Popen, track_as: str | None):
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_queue.put(f"[{entry_point}] {line}")
        proc.wait()
        self.log_queue.put(f"[{entry_point}] exited (code {proc.returncode})\n")
        if track_as is not None:
            self.log_queue.put(("status", track_as, "stopped", "grey"))
            self.processes[track_as] = None
            if track_as == "hoops":
                # Hoops session just ended — fresh stats are in shots.db,
                # and the in-game cooldown just got reset (though Idleon
                # may not flush the save to disk for a few more seconds).
                # Stamp client-side cooldown state immediately so the
                # display reflects the just-played state without waiting
                # for the save flush.
                try:
                    hoops_cooldown_state.write_set_now()
                except OSError as e:
                    self.log_queue.put(f"[launcher] couldn't write hoops cooldown state: {e}\n")
                self.log_queue.put(("refresh_hoops_stats",))
                self.log_queue.put(("refresh_tries",))
        if entry_point in self.setup_buttons:
            self.log_queue.put(("setup_done", entry_point))

    def _enqueue_log(self, text: str):
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
                        if entry_point in self.setup_buttons:
                            btn, label = self.setup_buttons[entry_point]
                            btn.config(state="normal", text=label)
                    elif item[0] == "refresh_hoops_stats":
                        self._refresh_hoops_stats()
                    elif item[0] == "refresh_tries":
                        self._refresh_tries(silent=True)
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


if __name__ == "__main__":
    run()
