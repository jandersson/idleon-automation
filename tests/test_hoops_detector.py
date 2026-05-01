"""Detector tests for hoops.

Pure-logic checks against synthetic frames built from the templates
themselves — verifies the function wires up correctly and returns
sensible coordinates. Doesn't depend on real game frames.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from minigames.hoops.detector import find_hoop_structure


HOOPS_ASSETS = Path(__file__).parent.parent / "minigames" / "hoops" / "assets"


def _frame_with_template_at(template_name: str, frame_w: int, frame_h: int,
                            place_at: tuple[int, int]) -> np.ndarray:
    """Build a BGRA frame with the given template embedded at place_at (top-left).

    Returns BGRA because that's what mss.grab gives detectors in production.
    """
    template = cv2.imread(str(HOOPS_ASSETS / template_name), cv2.IMREAD_COLOR)
    assert template is not None, f"missing {template_name}"
    th, tw = template.shape[:2]
    frame_bgr = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)
    x, y = place_at
    frame_bgr[y:y + th, x:x + tw] = template
    # Convert to BGRA — detector's first step is COLOR_BGRA2BGR.
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)


def test_find_hoop_structure_locates_template_center():
    template = cv2.imread(str(HOOPS_ASSETS / "hoop_structure.png"), cv2.IMREAD_COLOR)
    th, tw = template.shape[:2]
    # Place the template in the right half of a synthetic frame so the
    # detector's region restriction (right half) finds it.
    place_x, place_y = 600, 200
    frame = _frame_with_template_at("hoop_structure.png", 960, 572, (place_x, place_y))
    pos, conf = find_hoop_structure(frame)
    assert pos is not None, f"detection failed (conf={conf:.2f})"
    expected_cx = place_x + tw // 2
    expected_cy = place_y + th // 2
    assert abs(pos[0] - expected_cx) <= 1
    assert abs(pos[1] - expected_cy) <= 1
    assert conf > 0.95


def test_find_hoop_structure_rejects_blank_frame():
    frame = np.full((572, 960, 4), 30, dtype=np.uint8)
    pos, conf = find_hoop_structure(frame)
    assert pos is None
    assert conf < 0.7
