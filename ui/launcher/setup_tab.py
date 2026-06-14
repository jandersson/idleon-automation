"""Setup tab: per-minigame calibration / debug buttons.

Lives in its own tab so the Bots tab stays focused on starting/stopping;
setup tools are used rarely (mostly during initial calibration). Each
button shells out to its entry point as a one-shot via process.run_oneshot.
"""
from tkinter import ttk

from ui.launcher import config, process


def build(parent: ttk.Frame, app) -> None:
    for i, mg in enumerate(config.MINIGAMES):
        label = f"{mg.get('emoji', '')} {mg['name'].capitalize()}".strip()
        frame = ttk.LabelFrame(parent, text=label, padding=6)
        frame.grid(row=i, column=0, sticky="ew", padx=8, pady=4)
        for setup_label, cmd in mg["setup"]:
            btn = ttk.Button(frame, text=setup_label,
                             command=lambda c=cmd: process.run_oneshot(app, c))
            btn.pack(side="left", padx=2)
            app.setup_buttons[cmd] = (btn, setup_label)
    parent.columnconfigure(0, weight=1)
