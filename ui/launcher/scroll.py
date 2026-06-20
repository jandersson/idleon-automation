"""Mouse-wheel scrolling for the launcher's scrollable canvases.

Tk delivers ``<MouseWheel>`` to the widget under the pointer and does NOT
bubble it to that widget's parent. A content-filled ``Canvas`` (whose rows
sit on top of it via ``create_window``) therefore never sees the wheel — the
rows swallow it. The old fix used ``bind_all``, but that is application-global:
with more than one scrollable tab, whichever canvas bound last hijacks the
wheel for every tab (see #93).

Instead, bind a single router on the *toplevel* — which IS in every widget's
bindtags, so it catches wheel events over any descendant — and route each
event to whichever registered canvas the pointer most recently entered. One
``<Enter>`` per canvas updates the target; the router is bound once per
window. Windows-only launcher, so only ``<MouseWheel>`` (±120/notch) is
handled.
"""
from __future__ import annotations

import tkinter as tk

# Which registered canvas the wheel currently scrolls (the last one the
# pointer entered). Module-global on purpose: one active target at a time.
_active: dict = {"canvas": None}


def _wheel_units(delta: int) -> int:
    """Wheel delta (±120 per notch on Windows) → ``yview_scroll`` units.

    Wheel up sends +120 → scroll up (-1); wheel down sends -120 → scroll
    down (+1). Sub-notch deltas round toward 0 (no scroll), matching the
    prior behaviour.
    """
    return -int(delta / 120)


def _route(event):
    canvas = _active["canvas"]
    if canvas is not None and canvas.winfo_exists():
        canvas.yview_scroll(_wheel_units(event.delta), "units")
        return "break"
    return None


def bind_mousewheel(canvas: tk.Canvas) -> None:
    """Scroll ``canvas`` with the wheel while the pointer is over it or its
    content. Safe to call for several canvases — they share one router and
    won't fight over the wheel."""
    canvas.bind("<Enter>", lambda _e: _active.__setitem__("canvas", canvas), add="+")
    top = canvas.winfo_toplevel()
    if not getattr(top, "_wheel_router_bound", False):
        top._wheel_router_bound = True
        top.bind("<MouseWheel>", _route, add="+")
