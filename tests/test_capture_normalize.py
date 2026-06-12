"""HiDPI capture normalization (docs/macos_port.md section 2.2).

The invariant under test: whatever pixel density the OS hands back,
grab_region/grab_fullscreen return arrays at the requested logical
size. We test the pure resize helper with synthetic arrays — no live
captures.
"""
import ast
from pathlib import Path

import numpy as np

from common.capture import _normalize_size


def test_1x_frame_passes_through_untouched():
    frame = np.zeros((50, 100, 4), dtype=np.uint8)
    out = _normalize_size(frame, 100, 50)
    assert out is frame  # no copy, no resize


def test_2x_frame_downscaled_to_logical_size():
    # Synthetic Retina grab: requested 100x50, OS returned 200x100.
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    frame[:, 100:] = 255  # right half white
    out = _normalize_size(frame, 100, 50)
    assert out.shape == (50, 100, 4)
    # Content survives in logical coordinates: left half dark, right white.
    assert out[:, :48].mean() < 10
    assert out[:, 52:].mean() > 245


def test_fractional_scale_downscaled():
    # 1.5x fractional scaling (e.g. external monitor): 150x75 -> 100x50.
    frame = np.full((75, 150, 4), 128, dtype=np.uint8)
    out = _normalize_size(frame, 100, 50)
    assert out.shape == (50, 100, 4)
    assert abs(int(out.mean()) - 128) <= 1


def test_window_module_has_no_toplevel_pygetwindow_import():
    """pygetwindow's macOS backend breaks at import — the import must
    stay inside the Windows code path so `import common.window` is safe
    on every platform."""
    src = (Path(__file__).parent.parent / "common" / "window.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:  # module top level only
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        assert not any(n.startswith("pygetwindow") for n in names), (
            "pygetwindow imported at module top level — breaks macOS import"
        )
