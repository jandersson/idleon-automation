"""Render an annotated overlay video from a captured trace.

For each frame in a trace_<stamp>/ directory, runs the detector and
draws what it found — plank-top line, cart box, pit boxes, ore boxes,
plus the find_next_terrain result as text. Stitches into an MP4 so we
can visually QA the entire trace at once.

Usage:
    uv run mining-render-overlay <trace_dir>
    uv run mining-render-overlay trace_20260515_220235          # short form
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from minigames.mining.detector import (
    _find_plank_top_y,
    _find_plank_x_range,
    _scan_plank_ore,
    _scan_plank_pits,
    SCAN_BUFFER_PX,
    find_cart_detailed,
    find_next_terrain,
)

_HERE = Path(__file__).parent
CAPTURES_DIR = _HERE / "assets" / "captures"
ANALYSIS_DIR = _HERE / "assets" / "analysis"


def _resolve_trace_dir(arg: str) -> Path:
    p = Path(arg)
    if p.exists() and p.is_dir():
        return p
    # Try as a short form relative to captures/
    rel = CAPTURES_DIR / arg
    if rel.exists():
        return rel
    print(f"Trace dir not found: {arg}")
    sys.exit(1)


def _annotate(frame, plank_y, cart, cart_w, pits, ores, next_terrain, plank_range):
    """Draw all detector results on a copy of frame, return it."""
    out = frame.copy()
    h, w = out.shape[:2]
    if plank_y is not None:
        # Only draw plank line across the detected plank extent.
        if plank_range is not None:
            cv2.line(out, (plank_range[0], plank_y), (plank_range[1], plank_y),
                     (0, 200, 0), 1)
            # Faint vertical markers at the plank edges
            cv2.line(out, (plank_range[0], plank_y - 30), (plank_range[0], plank_y + 5),
                     (0, 150, 0), 1)
            cv2.line(out, (plank_range[1], plank_y - 30), (plank_range[1], plank_y + 5),
                     (0, 150, 0), 1)
        else:
            cv2.line(out, (0, plank_y), (w, plank_y), (0, 200, 0), 1)
    if cart is not None:
        half = cart_w // 2 if cart_w else 30
        # Cart box centered on the matched cart position. Cart aspect
        # ratio is ~2:1 (wide and short), so use half the width as the
        # height extent on each side.
        h_extent = max(15, half // 2)
        cv2.rectangle(out,
                      (cart[0] - half, cart[1] - h_extent),
                      (cart[0] + half, cart[1] + h_extent),
                      (0, 255, 0), 2)
        cv2.circle(out, cart, 3, (0, 255, 0), -1)
    if plank_y is not None:
        for a, b in pits:
            cv2.rectangle(out, (a, plank_y - 2), (b, plank_y + 8),
                          (0, 0, 255), 2)  # red
        for a, b in ores:
            cv2.rectangle(out, (a, plank_y - 18), (b, plank_y),
                          (0, 200, 255), 2)  # yellow
    label = f"plank_y={plank_y} cart={cart} next={next_terrain}"
    cv2.rectangle(out, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(out, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    return out


def _process_frame(frame):
    """Return (plank_y, cart, cart_w, pits, ores, next_terrain) for the
    frame — running the same logic find_next_terrain does so the overlay
    matches the bot's actual decision inputs."""
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    plank_y = _find_plank_top_y(frame)
    if plank_y is None:
        return None, None, 0, [], [], None, None
    det = find_cart_detailed(frame, plank_y=plank_y)
    cart = det["center"] if det else None
    cart_w = det["half_width"] * 2 if det else 0
    cart_right = (cart[0] + det["half_width"]) if det else None
    pits = []
    ores = []
    plank_range = _find_plank_x_range(frame, plank_y)
    if cart is not None:
        scan_x = cart_right + SCAN_BUFFER_PX
        x_end = plank_range[1] if plank_range else None
        pits = _scan_plank_pits(frame, plank_y, x_start=scan_x, x_end=x_end)
        ores = _scan_plank_ore(frame, plank_y, x_start=scan_x, x_end=x_end)
    nxt = find_next_terrain(frame, cart, plank_y=plank_y, cart_right=cart_right)
    return plank_y, cart, cart_w, pits, ores, nxt, plank_range


def _latest_trace() -> Path | None:
    traces = sorted(CAPTURES_DIR.glob("trace_*"))
    return traces[-1] if traces else None


def run():
    parser = argparse.ArgumentParser(description="Render annotated overlay video from a trace")
    parser.add_argument("trace_dir", nargs="?",
                        help="Path or short name (relative to captures/) of trace directory. "
                             "If omitted, uses the most recent trace_* dir.")
    parser.add_argument("--fps", type=int, default=10,
                        help="output video fps (default 10, matches 100ms capture)")
    args = parser.parse_args()

    if args.trace_dir:
        trace = _resolve_trace_dir(args.trace_dir)
    else:
        trace = _latest_trace()
        if trace is None:
            print("No trace dirs in captures/. Run mining-trace first.")
            sys.exit(1)
        print(f"Using most recent trace: {trace.name}")
    frame_paths = sorted(trace.glob("frame_*.png"))
    if not frame_paths:
        print(f"No frame_*.png in {trace}")
        sys.exit(1)

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{trace.name}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        print(f"VideoWriter failed to open {out_path}")
        sys.exit(1)

    print(f"Processing {len(frame_paths)} frames from {trace.name} -> {out_path.name}")
    for i, p in enumerate(frame_paths):
        frame = cv2.imread(str(p))
        plank_y, cart, cart_w, pits, ores, nxt, plank_range = _process_frame(frame)
        annotated = _annotate(frame, plank_y, cart, cart_w, pits, ores, nxt, plank_range)
        writer.write(annotated)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frame_paths)} frames")
    writer.release()
    print(f"Wrote {out_path} ({len(frame_paths)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    run()
