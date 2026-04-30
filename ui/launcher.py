"""Tkinter launcher for the Idleon bots.

One row per minigame with Start/Stop and inline setup-tool buttons. Each
button shells out to `uv run <entry-point>` in a subprocess; stdout is
streamed into the log pane at the bottom.
"""
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

PROJECT_ROOT = Path(__file__).parent.parent

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
        self.root.geometry("780x520")

        self.processes: dict[str, subprocess.Popen | None] = {m["name"]: None for m in MINIGAMES}
        self.status_labels: dict[str, ttk.Label] = {}
        self.log_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log_queue()

    def _build_ui(self):
        for i, mg in enumerate(MINIGAMES):
            frame = ttk.LabelFrame(self.root, text=mg["name"].capitalize(), padding=6)
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
                ttk.Button(tools, text=label,
                           command=lambda c=cmd: self._run_oneshot(c)).pack(side="left", padx=2)

        log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frame.grid(row=len(MINIGAMES), column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.log_text = tk.Text(log_frame, height=10, wrap="none", state="disabled",
                                bg="#111", fg="#ddd", insertbackground="#ddd",
                                font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(len(MINIGAMES), weight=1)

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
        # `uv run` spawns python as a child — kill the whole tree on Windows.
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
            # Detach from console so closing the launcher doesn't kill children.
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

    def _enqueue_log(self, text: str):
        self.log_queue.put(text)

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "status":
                    _, name, text, color = item
                    if name in self.status_labels:
                        self.status_labels[name].config(text=text, foreground=color)
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
