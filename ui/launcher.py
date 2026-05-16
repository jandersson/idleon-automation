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
from common.idleon_save import read_minigame_plays_shared, read_hoops_cooldown

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
            ("Pick game over", "chopping-pick-game-over"),
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
                "values": ["gp", "knn", "bivariate", "trajectory_knn", "trajectory_gp", "trajectory_rf"],
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
            ("Trace (record attempt)", "mining-trace"),
            ("Pick play region", "mining-pick-play-region"),
            ("Pick start button", "mining-pick-start-button"),
            ("Render overlay video", "mining-render-overlay"),
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
        # Hoops predictor comparison cards (Pokemon-style, one per kind).
        # Map predictor_kind -> dict of widget refs for fast updating.
        # Built in _build_bots_tab when the hoops row is constructed.
        self.predictor_cards: dict[str, dict] = {}
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
        sql_tab = ttk.Frame(nb)
        nb.add(bots_tab, text="Bots")
        nb.add(setup_tab, text="Setup")
        nb.add(frames_tab, text="Frames")
        nb.add(sql_tab, text="SQL")

        self._build_bots_tab(bots_tab)
        self._build_setup_tab(setup_tab)
        self._build_frames_tab(frames_tab)
        self._build_sql_tab(sql_tab)

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

            # Hoops gets a Pokemon-style stat-card row, one per predictor
            # — see GitHub issue #21. Each card shows Lv (sessions),
            # HP-bar (make rate), and shots/makes. Active predictor
            # (matching the dropdown) is highlighted. Other minigames
            # don't have a comparable signal yet.
            if mg["name"] == "hoops":
                self._build_predictor_cards(frame)
                self._refresh_hoops_stats()
                # Re-highlight the active card whenever the dropdown changes.
                pred_var = self.bot_option_vars.get((mg["name"], "HOOPS_PREDICTOR_KIND"))
                if pred_var is not None:
                    pred_var.trace_add("write", lambda *_a: self._highlight_active_predictor())
                self._highlight_active_predictor()

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

        # 🚧 = save-flush timing makes this unreliable; see GitHub #22.
        ttk.Label(strip, text="🏀 Cooldown 🚧:").pack(side="left", padx=(16, 4))
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

    # Predictor "cards" — one per kind in the launcher's hoops row.
    # Type colors are arbitrary but consistent (each model gets one).
    _PREDICTOR_CARD_SPECS = [
        ("gp",             "🧠", "GP",     "#7B68EE"),
        ("knn",            "🎯", "KNN",    "#4682B4"),
        ("bivariate",      "📐", "Biv",    "#808080"),
        ("trajectory_knn", "📍", "T-KNN",  "#FF8C00"),
        ("trajectory_gp",  "🌐", "T-GP",   "#DAA520"),
        ("trajectory_rf",  "🌲", "T-RF",   "#228B22"),
    ]

    def _build_predictor_cards(self, parent: ttk.Frame) -> None:
        """Build a horizontal row of stat cards, one per predictor kind."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))
        for kind, emoji, label, color in self._PREDICTOR_CARD_SPECS:
            # tk.Frame (not ttk) so we can set a per-card background and
            # a coloured border via highlightbackground for the "active"
            # outline. ttk.Frame styling is more painful for this.
            card = tk.Frame(
                row, relief="solid", borderwidth=1, padx=6, pady=4,
                highlightthickness=2, highlightbackground="#444",
                bg="#222",
            )
            card.pack(side="left", padx=2)
            name = tk.Label(
                card, text=f"{emoji} {label}", fg=color, bg="#222",
                font=("Segoe UI", 9, "bold"),
            )
            name.pack()
            lv = tk.Label(card, text="Lv 0", fg="#bbb", bg="#222",
                          font=("Segoe UI", 8))
            lv.pack()
            hp = ttk.Progressbar(card, length=80, mode="determinate", maximum=100)
            hp.pack(pady=(2, 1))
            rate = tk.Label(card, text="0/0", fg="#888", bg="#222",
                            font=("Consolas", 8))
            rate.pack()
            self.predictor_cards[kind] = {
                "card": card, "lv": lv, "hp": hp, "rate": rate,
                "color": color,
            }

    def _refresh_hoops_stats(self) -> None:
        """Pull per-predictor stats from shots.db and update each card.
        Cheap query (one GROUP BY over a few-hundred-row table); called
        on launcher start and whenever a hoops session ends."""
        if not self.predictor_cards:
            return
        import sqlite3
        db_path = MINIGAMES_DIR / "hoops" / "assets" / "shots.db"
        per_kind: dict[str, tuple[int, int, int]] = {}  # kind -> (sessions, shots, makes)
        if db_path.exists():
            try:
                con = sqlite3.connect(str(db_path))
                rows = con.execute(
                    "SELECT predictor_kind, COUNT(DISTINCT session_started), "
                    "COUNT(*), SUM(made) "
                    "FROM shots WHERE direction='up' AND predictor_kind IS NOT NULL "
                    "GROUP BY predictor_kind"
                ).fetchall()
                con.close()
                for kind, sessions, shots, makes in rows:
                    per_kind[kind] = (int(sessions), int(shots), int(makes or 0))
            except sqlite3.Error:
                pass  # leave cards in their default state
        for kind, widgets in self.predictor_cards.items():
            sessions, shots, makes = per_kind.get(kind, (0, 0, 0))
            rate_pct = int(round(100 * makes / shots)) if shots else 0
            widgets["lv"].config(text=f"Lv {sessions}")
            widgets["hp"]["value"] = rate_pct
            if shots:
                widgets["rate"].config(text=f"{makes}/{shots}  {rate_pct}%",
                                       fg="#ddd")
            else:
                widgets["rate"].config(text="—", fg="#666")

    def _highlight_active_predictor(self) -> None:
        """Outline the predictor card that matches the launcher's
        Predictor dropdown selection."""
        if not self.predictor_cards:
            return
        active = ""
        var = self.bot_option_vars.get(("hoops", "HOOPS_PREDICTOR_KIND"))
        if var is not None:
            active = var.get()
        for kind, widgets in self.predictor_cards.items():
            if kind == active:
                widgets["card"].config(highlightbackground=widgets["color"],
                                       highlightthickness=2)
            else:
                widgets["card"].config(highlightbackground="#444",
                                       highlightthickness=2)

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
        """Update the hoops cooldown label, extrapolating from the last
        save-read snapshot. Re-arms itself once a second.

        We can't predict the cooldown accurately ahead of a save flush
        (the duration appears to escalate with consecutive plays via
        OLA[424] and we don't have the formula yet), so we surface
        save-staleness as a "(save Xm old)" suffix and let the user
        force a flush via character switch when the displayed value
        matters.
        """
        import time
        mtime = self._save_mtime()
        if mtime > self.last_save_mtime:
            self.last_save_mtime = mtime
            self._refresh_tries(silent=True)
        snap = self.hoops_cooldown_snapshot
        if snap is None:
            self.hoops_label.config(text="—")
        else:
            save_at, stored_cd = snap
            now = time.time()
            remaining = stored_cd - (now - save_at)
            save_age = now - save_at
            stale = save_age > 30
            if remaining <= 0:
                # Don't claim "ready" off a stale snapshot — the user
                # may have played in-game since the last flush, in
                # which case there's an active cooldown we can't see.
                # Verified 2026-05-10 14:49: in-game showed 219s while
                # launcher said "ready (save 1m32s old)" — misleading.
                base = "?" if stale else "ready"
            else:
                m, s = divmod(int(remaining) + 1, 60)
                base = f"{m}:{s:02d}"
            if stale:
                am, asec = divmod(int(save_age), 60)
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
        plays = read_minigame_plays_shared()
        if plays is None:
            self.tries_label.config(text="(save not found)")
            if not silent:
                self._enqueue_log("[tries] couldn't read save (Idleon not installed, or plyvel missing)\n")
            return
        self.tries_label.config(text=f"{plays}")
        # Persist for any external consumer.
        tries_counter.write(plays)
        if not silent:
            self._enqueue_log(f"[tries] refreshed: {plays}\n")

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
                # and the in-game cooldown just got reset. We don't know
                # the new cooldown's duration (it appears to escalate
                # with consecutive plays via OLA[424]); rely on the
                # save-mtime watcher to pick up the real value as soon
                # as Idleon flushes.
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

    def _build_sql_tab(self, parent: ttk.Frame):
        """Lightweight read-only query browser over the per-bot SQLite DBs.

        Top row: DB picker + Run button. Middle: query editor pre-seeded
        with a useful starting query for the selected DB. Bottom:
        results table (column headers from the cursor description)."""
        # Discover available DBs by scanning minigames/<bot>/assets/*.db
        dbs: dict[str, Path] = {}
        for mg_dir in sorted(MINIGAMES_DIR.iterdir()):
            if not mg_dir.is_dir():
                continue
            for db_path in (mg_dir / "assets").glob("*.db"):
                dbs[f"{mg_dir.name} / {db_path.name}"] = db_path
        if not dbs:
            ttk.Label(parent, text="No DBs found under minigames/*/assets/.").pack(pady=20)
            return

        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="DB:").pack(side="left")
        db_var = tk.StringVar(value=next(iter(dbs)))
        db_combo = ttk.Combobox(top, textvariable=db_var, state="readonly",
                                values=list(dbs.keys()), width=40)
        db_combo.pack(side="left", padx=(4, 12))
        ttk.Button(top, text="Run (Ctrl+Enter)", width=18,
                   command=lambda: self._sql_run(dbs[db_var.get()],
                                                 query_text.get("1.0", "end").strip(),
                                                 tree, status)).pack(side="left")
        status = ttk.Label(top, text="", foreground="grey")
        status.pack(side="left", padx=(12, 0))

        # Query editor
        query_frame = ttk.LabelFrame(parent, text="Query", padding=4)
        query_frame.pack(fill="x", padx=8, pady=4)
        query_text = tk.Text(query_frame, height=8, wrap="none",
                             font=("Consolas", 10))
        query_text.pack(fill="x")
        query_text.bind("<Control-Return>",
                        lambda e: (self._sql_run(dbs[db_var.get()],
                                                 query_text.get("1.0", "end").strip(),
                                                 tree, status), "break")[1])

        # When DB changes, reset the query to a sensible starter
        def _on_db_change(*_):
            query_text.delete("1.0", "end")
            query_text.insert("1.0", self._sql_starter_query(dbs[db_var.get()]))
        db_var.trace_add("write", _on_db_change)
        _on_db_change()

        # Results table — Treeview with dynamic columns built per query
        results_frame = ttk.LabelFrame(parent, text="Results", padding=4)
        results_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        tree = ttk.Treeview(results_frame, show="headings")
        ysb = ttk.Scrollbar(results_frame, orient="vertical", command=tree.yview)
        xsb = ttk.Scrollbar(results_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)

    def _sql_starter_query(self, db_path: Path) -> str:
        """Return a sensible read-only starter query for the given DB.
        Tailored per-bot since the schemas differ; falls back to a
        generic 'show schema'."""
        name = db_path.name
        if name == "mining.db":
            return (
                "-- Survival rate per pit-distance bin\n"
                "SELECT (next_distance_px / 10) * 10 AS dist_bin,\n"
                "       COUNT(*) AS jumps,\n"
                "       SUM(CASE WHEN outcome = 'survived' THEN 1 ELSE 0 END) AS survived,\n"
                "       SUM(CASE WHEN outcome = 'died' THEN 1 ELSE 0 END) AS died\n"
                "FROM jumps\n"
                "WHERE next_kind = 'pit' AND outcome IS NOT NULL\n"
                "GROUP BY dist_bin\n"
                "ORDER BY dist_bin"
            )
        if name == "shots.db":
            return (
                "-- Recent hoops shots\n"
                "SELECT fired_at, hoop_x, hoop_y, \"offset\", made, score_diff, direction\n"
                "FROM shots\n"
                "ORDER BY id DESC LIMIT 50"
            )
        if name == "darts.db":
            return (
                "-- Recent darts throws\n"
                "SELECT fired_at, launch_angle_deg, landing_x, hit, bullseye, streak\n"
                "FROM throws\n"
                "ORDER BY id DESC LIMIT 50"
            )
        return "SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'view')"

    def _sql_run(self, db_path: Path, sql: str,
                 tree: ttk.Treeview, status: ttk.Label) -> None:
        """Execute the query against the selected DB and load results
        into the tree. Read-only enforced via sqlite3 query_only PRAGMA —
        no DDL or DML can run from this tab."""
        import sqlite3
        if not sql:
            status.config(text="empty query", foreground="grey")
            return
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = ON")
            cur = conn.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        except sqlite3.Error as e:
            status.config(text=f"error: {e}", foreground="red")
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # Reset tree
        for c in tree["columns"]:
            tree.heading(c, text="")
        tree.delete(*tree.get_children())
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(80, min(300, len(c) * 12)), anchor="w")
        for r in rows:
            tree.insert("", "end", values=tuple("" if v is None else v for v in r))
        status.config(text=f"{len(rows)} row(s)", foreground="green")

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
