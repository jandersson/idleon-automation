"""Subprocess machinery: start/stop bots and one-shot setup tools.

Each bot/tool runs as `uv run --no-sync <entry-point>`; stdout is drained
on a background thread into app.log_queue, which the app shell's poll loop
renders. Functions take the Launcher app as their first argument and read
its shared registries (processes, status_labels, setup_buttons,
bot_option_vars).
"""
import os
import subprocess
import sys
import threading

from ui.launcher import theme
from ui.launcher.config import PROJECT_ROOT


def start_bot(app, mg: dict) -> None:
    name = mg["name"]
    if app.processes.get(name) is not None:
        app.enqueue_log(f"[{name}] already running\n")
        return
    # Mutual exclusion: bots drive a single mouse/keyboard via pyautogui.
    # Running two at once produced (on 2026-05-24) darts throws getting
    # fired mid-arm-swing by hoops's platform clicks, plus polluted DB
    # rows in both bots. Refuse the launch and tell the user which bot
    # is holding the slot so they can stop it first.
    already_running = [n for n, p in app.processes.items() if p is not None]
    if already_running:
        holder = already_running[0]
        app.enqueue_log(
            f"[{name}] not started — {holder} is running. "
            f"Stop {holder} first; only one bot at a time.\n"
        )
        return
    extra_env: dict[str, str] = {}
    for opt in mg.get("bot_options", []):
        var = app.bot_option_vars.get((name, opt["env"]))
        if var is not None:
            extra_env[opt["env"]] = var.get()
    if extra_env:
        kv = ", ".join(f"{k}={v}" for k, v in extra_env.items())
        app.enqueue_log(f"[{name}] options: {kv}\n")
    spawn(app, mg["bot"], track_as=name, extra_env=extra_env)
    app.status_labels[name].config(text="running", foreground=theme.SUCCESS)


def stop_bot(app, mg: dict) -> None:
    name = mg["name"]
    proc = app.processes.get(name)
    if proc is None:
        return
    app.enqueue_log(f"[{name}] stopping...\n")
    _kill(proc)


def stop_oneshot(app, entry_point: str) -> None:
    """Stop a running setup/observe tool (observe, capture, watch-wind have
    no other GUI stop). The button restores itself when the process exits,
    via the setup_done queue message."""
    proc = app.oneshot_procs.get(entry_point)
    if proc is None:
        return
    app.enqueue_log(f"[{entry_point}] stopping...\n")
    _kill(proc)


def _kill(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.terminate()


def run_oneshot(app, cmd: str) -> None:
    spawn(app, cmd, track_as=None)


def spawn(app, entry_point: str, track_as: str | None,
          extra_env: dict[str, str] | None = None) -> None:
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
        app.enqueue_log(f"[{entry_point}] could not run — is `uv` on your PATH?\n")
        return
    if track_as is not None:
        app.processes[track_as] = proc
    if entry_point in app.setup_buttons:
        # Setup tools are launched as one-shots (track_as=None), but some
        # run until stopped (observe, capture, watch-wind). Turn the button
        # into a Stop while it runs so there's a GUI control — without this
        # there was no way to stop observe from the launcher. Quick tools
        # (pickers) just exit and restore the button. setup_done restores it.
        app.oneshot_procs[entry_point] = proc
        btn, label = app.setup_buttons[entry_point]
        btn.config(state="normal", text=f"■ Stop {label}", style="Stop.TButton",
                   command=lambda e=entry_point: stop_oneshot(app, e))
    app.enqueue_log(f"[{entry_point}] started (pid {proc.pid})\n")
    threading.Thread(target=drain, args=(app, entry_point, proc, track_as),
                     daemon=True).start()


def drain(app, entry_point: str, proc: subprocess.Popen, track_as: str | None) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        app.log_queue.put(f"[{entry_point}] {line}")
    proc.wait()
    app.log_queue.put(f"[{entry_point}] exited (code {proc.returncode})\n")
    if track_as is not None:
        app.log_queue.put(("status", track_as, "stopped", theme.MUTED))
        app.processes[track_as] = None
        if track_as == "hoops":
            # Hoops session just ended — fresh stats are in shots.db,
            # and the in-game cooldown just got reset. We don't know
            # the new cooldown's duration (it appears to escalate
            # with consecutive plays via OLA[424]); rely on the
            # save-mtime watcher to pick up the real value as soon
            # as Idleon flushes.
            app.log_queue.put(("refresh_hoops_stats",))
            app.log_queue.put(("refresh_tries",))
    if entry_point in app.setup_buttons:
        app.log_queue.put(("setup_done", entry_point))
