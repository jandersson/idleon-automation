"""Tests for the mining live digit-capture path (#6).

Synthetic crops only — verifies the dedup signature and that the
DigitCapturer writes one PNG per distinct rendered number plus a
manifest, the contract the offline digit-template bootstrap relies on.
"""
import json

import numpy as np

from minigames.mining.digit_capture import DigitCapturer, crop_signature

# Dark red-brown background like a real score region; white foreground.
DARK = (0, 16, 58)
FG = (249, 249, 249)


def _crop_with_block(w_block: int, h_block: int) -> np.ndarray:
    crop = np.full((22, 30, 3), DARK, dtype=np.uint8)
    crop[5:5 + h_block, 5:5 + w_block] = FG
    return crop


def test_crop_signature_none_on_blank():
    blank = np.full((22, 30, 3), DARK, dtype=np.uint8)
    assert crop_signature(blank) is None


def test_crop_signature_stable_for_identical_renders():
    sig = crop_signature(_crop_with_block(6, 9))
    assert sig is not None
    assert sig == crop_signature(_crop_with_block(6, 9))


def test_crop_signature_differs_for_different_renders():
    assert crop_signature(_crop_with_block(6, 9)) != crop_signature(_crop_with_block(9, 9))


def test_capturer_dedupes_and_writes_crops(tmp_path):
    cap = DigitCapturer(tmp_path)
    assert cap.maybe_save(_crop_with_block(6, 9)) is True
    assert cap.maybe_save(_crop_with_block(6, 9)) is False  # identical → deduped
    assert cap.maybe_save(_crop_with_block(9, 9)) is True
    assert cap.count == 2
    assert (tmp_path / "crop_000.png").exists()
    assert (tmp_path / "crop_001.png").exists()


def test_capturer_flush_writes_manifest(tmp_path):
    cap = DigitCapturer(tmp_path)
    cap.maybe_save(_crop_with_block(6, 9), frame_idx=12, t_rel=1.5)
    manifest = cap.flush()
    assert manifest is not None and manifest.exists()
    data = json.loads(manifest.read_text())
    assert len(data["crops"]) == 1
    entry = data["crops"][0]
    assert entry["crop"] == "crop_000.png"
    assert entry["frame_idx"] == 12
    assert entry["n_components"] == 1


def test_capturer_flush_none_when_empty(tmp_path):
    assert DigitCapturer(tmp_path).flush() is None


def test_capturer_ignores_none_and_blank(tmp_path):
    cap = DigitCapturer(tmp_path)
    assert cap.maybe_save(None) is False
    assert cap.maybe_save(np.full((22, 30, 3), DARK, dtype=np.uint8)) is False
    assert cap.count == 0


def test_capturer_respects_max_crops(tmp_path):
    cap = DigitCapturer(tmp_path, max_crops=1)
    assert cap.maybe_save(_crop_with_block(6, 9)) is True
    assert cap.maybe_save(_crop_with_block(9, 9)) is False  # cap reached
    assert cap.count == 1
