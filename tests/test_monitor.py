"""Tests for common/monitor.py — per-shot folder helpers."""
import re

import cv2
import numpy as np

from common.monitor import make_shot_dir, save_frame, save_meta


def test_make_shot_dir_creates_subfolder(tmp_path):
    sub = make_shot_dir(tmp_path, 1)
    assert sub.parent == tmp_path
    assert sub.exists()
    assert sub.is_dir()


def test_make_shot_dir_filename_pattern(tmp_path):
    sub = make_shot_dir(tmp_path, 7, prefix="throw")
    # Format: <prefix>_<NNN>_<HHMMSS>
    assert re.fullmatch(r"throw_007_\d{6}", sub.name)


def test_make_shot_dir_default_prefix(tmp_path):
    sub = make_shot_dir(tmp_path, 42)
    assert sub.name.startswith("shot_042_")


def test_save_frame_converts_bgra_to_bgr(tmp_path):
    bgra = np.zeros((10, 20, 4), dtype=np.uint8)
    bgra[..., 0] = 255  # blue channel max
    bgra[..., 3] = 255  # alpha
    target = tmp_path / "frame.png"
    save_frame(target, bgra)

    assert target.exists()
    saved = cv2.imread(str(target), cv2.IMREAD_COLOR)
    assert saved.shape == (10, 20, 3)  # 3 channels (BGR), not 4
    # Blue channel should still be 255 in the saved BGR image.
    assert saved[5, 10, 0] == 255


def test_save_meta_writes_key_value_lines(tmp_path):
    target = tmp_path / "meta.txt"
    save_meta(target, hoop="(100,200)", offset=15, made=True)
    text = target.read_text(encoding="utf-8")
    assert "hoop=(100,200)" in text
    assert "offset=15" in text
    assert "made=True" in text


def test_save_meta_one_field_per_line(tmp_path):
    target = tmp_path / "meta.txt"
    save_meta(target, a=1, b=2, c=3)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
