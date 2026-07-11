"""Subprocess machinery: start/stop bots and one-shot setup tools.

Each bot/tool runs as `uv run --no-sync <entry-point>`; stdout is drained
on a background thread into the app's event queue, which the shell's poll
loop dispatches on the Tk thread (see app.Launcher.EVENTS). This module
never touches widgets directly — it emits events; the tabs subscribe.

Events emitted (all via app.emit, thread-safe):
    log(text)                — one log line
    bot_started(name)        — a tracked bot began
    bot_exited(name, code)   — a tracked bot ended
    tool_started(entry)      — a setup/observe tool began
    tool_exited(entry)       — a setup/observe tool ended
"""
import os
import subprocess
import sys
import threading

from ui.launcher.config import PROJECT_ROOT


def running_bot(app) -> str | None:
    """Name of the currently-running bot, or None. Bots are mutually
    exclusive: they drive a single mouse/keyboard via pyautogui — two at
    once produced cross-fired clicks and polluted DB rows (2026-05-24)."""
    for name, proc in app.processes.items():
        if proc is not None:
            return name
    return None


def start_bot(app, mg: dict) -> None:
    name = mg["name"]
    holder = running_bot(app)
    if holder is not None:
        # The UI disables Start buttons while a bot runs, so this is a
        # race backstop, not the primary guard.
        app.emit("log", f"[{name}] not started — {holder} is running\n")
        return
    extra_env: dict[str, str] = {}
    for opt in mg.get("bot_options", []):
        var = app.bot_option_vars.get((name, opt["env"]))
        if var is not None:
            extra_env[opt["env"]] = var.get()
    if extra_env:
        kv = ", ".join(f"{k}={v}" for k, v in extra_env.items())
        app.emit("log", f"[{name}] options: {kv}\n")
    _spawn(app, mg["bot"], track_as=name, extra_env=extra_env)


def stop_bot(app, mg: dict) -> None:
    proc = app.processes.get(mg["name"])
    if proc is None:
        return
    app.emit("log", f"[{mg['name']}] stopping...\n")
    _kill(proc)


def run_oneshot(app, entry_point: str) -> None:
    _spawn(app, entry_point, track_as=None)


def stop_oneshot(app, entry_point: str) -> None:
    """Stop a running setup/observe tool (observe, capture, watch-wind have
    no other GUI stop control)."""
    proc = app.oneshot_procs.get(entry_point)
    if proc is None:
        return
    app.emit("log", f"[{entry_point}] stopping...\n")
    _kill(proc)


def _kill(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.terminate()


def _spawn(app, entry_point: str, track_as: str | None,
           extra_env: dict[str, str] | None = None) -> None:
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW
    # PYTHONUNBUFFERED=1 so the bot's stdout flushes line-by-line rather
    # than block-buffering — keeps the launcher log live.
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
        app.emit("log", f"[{entry_point}] could not run — is `uv` on your PATH?\n")
        return
    if track_as is not None:
        app.processes[track_as] = proc
        app.emit("bot_started", track_as)
    else:
        app.oneshot_procs[entry_point] = proc
        app.emit("tool_started", entry_point)
    app.emit("log", f"[{entry_point}] started (pid {proc.pid})\n")
    threading.Thread(target=_drain, args=(app, entry_point, proc, track_as),
                     daemon=True).start()


def _drain(app, entry_point: str, proc: subprocess.Popen,
           track_as: str | None) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        app.emit("log", f"[{entry_point}] {line}")
    proc.wait()
    app.emit("log", f"[{entry_point}] exited (code {proc.returncode})\n")
    if track_as is not None:
        app.processes[track_as] = None
        app.emit("bot_exited", track_as, proc.returncode)
    else:
        app.oneshot_procs[entry_point] = None
        app.emit("tool_exited", entry_point)
