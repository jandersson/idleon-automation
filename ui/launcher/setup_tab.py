"""Setup tab: per-minigame calibration / debug tools.

Lives in its own tab so the Bots tab stays focused on starting/stopping.
Each button shells out to its entry point via process.run_oneshot; the
long-running tools (observe, capture, watch-wind) flip their button into
a Stop while running (driven by the tool_started/tool_exited events).
Buttons wrap into a grid instead of one overflowing row.
"""
import tkinter as tk
from tkinter import ttk

from ui.launcher import config, process, theme

_COLS = 5  # buttons per row inside a card


def build(parent: ttk.Frame, app) -> None:
    for mg in config.MINIGAMES:
        color = theme.bot_color(mg["name"])
        outer = tk.Frame(parent, bg=theme.SURFACE, highlightthickness=1,
                         highlightbackground=theme.BORDER)
        outer.pack(fill="x", padx=theme.PAD, pady=theme.PAD_XS)
        tk.Frame(outer, bg=color, width=4).pack(side="left", fill="y")
        body = ttk.Frame(outer, style="Card.TFrame",
                         padding=(theme.PAD, theme.PAD_S))
        body.pack(side="left", fill="both", expand=True)
        tk.Label(body, text=f"{mg.get('emoji', '')} {mg['name'].capitalize()}",
                 fg=color, bg=theme.SURFACE, font=theme.FONT_BOLD,
                 anchor="w").grid(row=0, column=0, columnspan=_COLS, sticky="w",
                                  pady=(0, theme.PAD_XS))
        for i, (setup_label, cmd) in enumerate(mg["setup"]):
            btn = ttk.Button(body, text=setup_label,
                             command=lambda c=cmd: process.run_oneshot(app, c))
            btn.grid(row=1 + i // _COLS, column=i % _COLS, sticky="ew",
                     padx=theme.PAD_XS, pady=theme.PAD_XS)
            app.setup_buttons[cmd] = (btn, setup_label)
        for c in range(_COLS):
            body.columnconfigure(c, weight=1)

    app.on("tool_started", lambda entry: _on_tool_started(app, entry))
    app.on("tool_exited", lambda entry: _on_tool_exited(app, entry))


def _on_tool_started(app, entry_point: str) -> None:
    """Turn the tool's launch button into a Stop while it runs — observe/
    capture/watch-wind have no other GUI stop control. Quick tools
    (pickers) exit fast and restore almost immediately."""
    if entry_point in app.setup_buttons:
        btn, label = app.setup_buttons[entry_point]
        btn.config(text=f"■ Stop {label}", style="Stop.TButton",
                   command=lambda e=entry_point: process.stop_oneshot(app, e))


def _on_tool_exited(app, entry_point: str) -> None:
    if entry_point in app.setup_buttons:
        btn, label = app.setup_buttons[entry_point]
        btn.config(state="normal", text=label, style="TButton",
                   command=lambda c=entry_point: process.run_oneshot(app, c))
