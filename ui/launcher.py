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
            ("Pick game over", "darts-pick-game-over"),
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
        frames_tab = ttk.Frame(nb)
        nb.add(bots_tab, text="Bots")
        nb.add(frames_tab, text="Frames")

        self._build_bots_tab(bots_tab)
        self._build_frames_tab(frames_tab)

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
