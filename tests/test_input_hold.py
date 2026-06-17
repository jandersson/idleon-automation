"""Tests for common.input.hold — the fishing cast press-and-hold primitive."""
import pytest

import common.input as inp


class _FakePyAutoGUI:
    def __init__(self):
        self.calls = []

    def moveTo(self, x, y):
        self.calls.append(("moveTo", x, y))

    def mouseDown(self):
        self.calls.append(("mouseDown",))

    def mouseUp(self):
        self.calls.append(("mouseUp",))


def test_hold_sequences_down_sleep_up(monkeypatch):
    fake = _FakePyAutoGUI()
    monkeypatch.setattr(inp, "pyautogui", fake)
    monkeypatch.setattr(inp.time, "sleep", lambda s: fake.calls.append(("sleep", s)))
    inp.hold(100, 200, 500, jitter=0)
    assert [c[0] for c in fake.calls] == ["moveTo", "mouseDown", "sleep", "mouseUp"]
    assert fake.calls[0] == ("moveTo", 100, 200)
    assert fake.calls[2] == ("sleep", 0.5)  # ms -> seconds


def test_hold_releases_even_if_sleep_raises(monkeypatch):
    fake = _FakePyAutoGUI()
    monkeypatch.setattr(inp, "pyautogui", fake)

    def boom(_s):
        raise KeyboardInterrupt()

    monkeypatch.setattr(inp.time, "sleep", boom)
    with pytest.raises(KeyboardInterrupt):
        inp.hold(0, 0, 100, jitter=0)
    assert ("mouseUp",) in fake.calls  # finally released the button


def test_hold_negative_duration_clamped_to_zero(monkeypatch):
    fake = _FakePyAutoGUI()
    monkeypatch.setattr(inp, "pyautogui", fake)
    captured = {}
    monkeypatch.setattr(inp.time, "sleep", lambda s: captured.setdefault("s", s))
    inp.hold(0, 0, -50, jitter=0)
    assert captured["s"] == 0
