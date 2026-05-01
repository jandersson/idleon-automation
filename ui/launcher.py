"""Tkinter launcher for the Idleon bots.

Two tabs:
- Bots: per-minigame Start/Stop and inline setup-tool buttons. Each button
  shells out to `uv run <entry-point>` in a subprocess; stdout streams into
  the log pane at the bottom.
- Shots: image inspector for per-shot monitor folders. Pick a minigame +
  shot folder, see the saved pre/post/flight frames stacked.
"""
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).parent.parent
MINIGAMES_DIR = PROJECT_ROOT / "minigames"

MINIGAMES = [
    {
        "name": "chopping",
        "bot": "chopping",
        "setup": [
            ("Pick bar", "chopping-pick-bar-region"),
            ("Pick leaf", "chopping-pick-leaf-region"),
            ("Pick button", "chopping-pick-button-region"),
            ("Calibrate", "chopping-calibrate"),
        ],
    },
    {
        "name": "hoops",
        "bot": "hoops",
        "setup": [
            ("Capture", "hoops-capture"),
            ("Debug match", "hoops-debug"),
            ("Ball calibrate", "hoops-ball-calibrate"),
            ("Score calibrate", "hoops-score-calibrate"),
            ("Pick score", "hoops-pick-score-region"),
            ("Pick game over", "hoops-pick-game-over"),
            ("Pick lives", "hoops-pick-lives-region"),
        ],
    },
    {
        "name": "darts",
        "bot": "darts",
        "setup": [
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
        "bot": "catching",
        "setup": [
            ("Capture", "catching-capture"),
            ("Pick play region", "catching-pick-play-region"),
            ("Extract fly", "catching-extract-fly"),
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

        # Shots tab state — keep PhotoImage refs alive so Tk doesn't GC them.
        self.shot_images: list[ImageTk.PhotoImage] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log_queue()

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        bots_tab = ttk.Frame(nb)
        shots_tab = ttk.Frame(nb)
        nb.add(bots_tab, text="Bots")
        nb.add(shots_tab, text="Shots")

        self._build_bots_tab(bots_tab)
        self._build_shots_tab(shots_tab)

    def _build_bots_tab(self, parent: ttk.Frame):
        for i, mg in enumerate(MINIGAMES):
            frame = ttk.LabelFrame(parent, text=mg["name"].capitalize(), padding=6)
            frame.grid(row=i, column=0, sticky="ew", padx=8, pady=4)

            top = ttk.Frame(frame)
            top.pack(fill="x")
            ttk.Button(top, text="Start", width=8,
                       command=lambda m=mg: self._start_bot(m)).pack(side="left")
            ttk.Button(top, text="Stop", width=8,
                       command=lambda m=mg: self._stop_bot(m)).pack(side="left", padx=(4, 12))
            status = ttk.Label(top, text="stopped", foreground="grey")
            status.pack(side="left")
            self.status_labels[mg["name"]] = status

            tools = ttk.Frame(frame)
            tools.pack(fill="x", pady=(4, 0))
            ttk.Label(tools, text="Setup:", foreground="grey").pack(side="left", padx=(0, 4))
            for label, cmd in mg["setup"]:
                btn = ttk.Button(tools, text=label,
                                 command=lambda c=cmd: self._run_oneshot(c))
                btn.pack(side="left", padx=2)
                self.setup_buttons[cmd] = (btn, label)

        log_frame = ttk.LabelFrame(parent, text="Log", padding=4)
        log_frame.grid(row=len(MINIGAMES), column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.log_text = tk.Text(log_frame, height=10, wrap="none", state="disabled",
                                bg="#111", fg="#ddd", insertbackground="#ddd",
                                font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(len(MINIGAMES), weight=1)

    def _build_shots_tab(self, parent: ttk.Frame):
        controls = ttk.Frame(parent, padding=6)
        controls.pack(fill="x")

        ttk.Label(controls, text="Minigame:").pack(side="left")
        self.shot_minigame = tk.StringVar(value="hoops")
        mg_picker = ttk.Combobox(
            controls, textvariable=self.shot_minigame, state="readonly",
            values=[mg["name"] for mg in MINIGAMES if (MINIGAMES_DIR / mg["name"] / "assets" / "monitor").exists()],
            width=12,
        )
        mg_picker.pack(side="left", padx=(4, 12))
        mg_picker.bind("<<ComboboxSelected>>", lambda _e: self._refresh_shots_list())

        ttk.Button(controls, text="Refresh", command=self._refresh_shots_list).pack(side="left")
        self.shot_status_label = ttk.Label(controls, text="", foreground="grey")
        self.shot_status_label.pack(side="left", padx=12)

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Left: shot folder list
        list_frame = ttk.LabelFrame(body, text="Shot folders", padding=4)
        list_frame.pack(side="left", fill="y", padx=(0, 6))
        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side="right", fill="y")
        self.shot_listbox = tk.Listbox(list_frame, width=28, height=20,
                                       yscrollcommand=list_scroll.set,
                                       font=("Consolas", 9))
        self.shot_listbox.pack(side="left", fill="y")
        list_scroll.config(command=self.shot_listbox.yview)
        self.shot_listbox.bind("<<ListboxSelect>>", lambda _e: self._show_selected_shot())

        # Right: scrollable image canvas
        viewer_frame = ttk.LabelFrame(body, text="Frames", padding=4)
        viewer_frame.pack(side="left", fill="both", expand=True)

        self.shot_canvas = tk.Canvas(viewer_frame, bg="#222", highlightthickness=0)
        self.shot_canvas.pack(side="left", fill="both", expand=True)
        canvas_scroll = ttk.Scrollbar(viewer_frame, orient="vertical",
                                      command=self.shot_canvas.yview)
        canvas_scroll.pack(side="right", fill="y")
        self.shot_canvas.config(yscrollcommand=canvas_scroll.set)

        self.shot_inner = ttk.Frame(self.shot_canvas)
        self.shot_canvas.create_window((0, 0), window=self.shot_inner, anchor="nw")
        self.shot_inner.bind(
            "<Configure>",
            lambda _e: self.shot_canvas.configure(scrollregion=self.shot_canvas.bbox("all")),
        )
        # Mouse wheel scroll on Windows.
        self.shot_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.shot_canvas.yview_scroll(-int(e.delta / 120), "units"),
        )

        self._refresh_shots_list()

    def _refresh_shots_list(self):
        self.shot_listbox.delete(0, "end")
        mg = self.shot_minigame.get()
        monitor = MINIGAMES_DIR / mg / "assets" / "monitor"
        if not monitor.exists():
            self.shot_status_label.config(text=f"no monitor dir for {mg}")
            return
        folders = sorted(
            (p for p in monitor.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in folders:
            self.shot_listbox.insert("end", p.name)
        self.shot_status_label.config(text=f"{len(folders)} shot folder(s)")

    def _show_selected_shot(self):
        sel = self.shot_listbox.curselection()
        if not sel:
            return
        folder_name = self.shot_listbox.get(sel[0])
        mg = self.shot_minigame.get()
        folder = MINIGAMES_DIR / mg / "assets" / "monitor" / folder_name
        if not folder.exists():
            return

        # Clear previous images
        for child in self.shot_inner.winfo_children():
            child.destroy()
        self.shot_images.clear()

        # Show meta.txt if present
        meta = folder / "meta.txt"
        if meta.exists():
            meta_text = tk.Text(self.shot_inner, height=4, wrap="word",
                                bg="#1a1a1a", fg="#ddd", font=("Consolas", 9))
            meta_text.insert("1.0", meta.read_text())
            meta_text.config(state="disabled")
            meta_text.pack(fill="x", padx=4, pady=(0, 6))

        # Order: pre, post, then flight frames in order
        png_files = list(folder.glob("*.png"))
        ordered = (
            [p for p in png_files if p.name == "pre_shot.png"]
            + [p for p in png_files if p.name == "post_shot.png"]
            + sorted(p for p in png_files if p.name.startswith("flight_"))
            + [p for p in png_files
               if p.name not in {"pre_shot.png", "post_shot.png"}
               and not p.name.startswith("flight_")]
        )
        for path in ordered:
            try:
                img = Image.open(path)
            except Exception as e:
                ttk.Label(self.shot_inner, text=f"{path.name}: {e}",
                          foreground="red").pack(anchor="w")
                continue
            # Cap width so the row of frames stays visible.
            max_w = 720
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)),
                                 Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.shot_images.append(photo)
            row = ttk.Frame(self.shot_inner)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=path.name, width=18,
                      foreground="#9cf").pack(side="left", anchor="n")
            tk.Label(row, image=photo, bd=0).pack(side="left")

    def _start_bot(self, mg: dict):
        name = mg["name"]
        if self.processes.get(name) is not None:
            self._enqueue_log(f"[{name}] already running\n")
            return
        self._spawn(mg["bot"], track_as=name)
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

    def _spawn(self, entry_point: str, track_as: str | None):
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(
                ["uv", "run", entry_point],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                creationflags=creationflags,
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
