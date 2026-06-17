"""Tests for common.input closed-loop charge-and-release (fishing cast).

`_charge_step` is the pure poll decision (tested exhaustively); the IO wrapper
`charge_and_release` is exercised with a fake pyautogui + a controllable clock,
mirroring tests/test_input_hold.py's fake-pyautogui style.
"""
import pytest

import common.input as inp


# --- _charge_step (pure) ---------------------------------------------------

def test_charge_step_release_when_target_reached():
    assert inp._charge_step(40, 0.1, True, 40, 0.3, 1.5) == "release"
    assert inp._charge_step(41, 0.1, True, 40, 0.3, 1.5) == "release"


def test_charge_step_release_when_max_hold_elapsed():
    # below target but the hold cap is hit (bar saturated short of target) -> cast
    assert inp._charge_step(55, 1.5, True, 99, 0.3, 1.5) == "release"


def test_charge_step_abort_when_not_ready_past_grace():
    # fill stayed 0 through the grace window -> rod not ready
    assert inp._charge_step(0, 0.3, False, 40, 0.3, 1.5) == "abort"
    assert inp._charge_step(0, 0.5, False, 40, 0.3, 1.5) == "abort"


def test_charge_step_holds_within_grace_even_at_zero():
    assert inp._charge_step(0, 0.1, False, 40, 0.3, 1.5) == "hold"


def test_charge_step_seen_positive_suppresses_abort_on_transient_zero():
    # once it has charged, a transient 0 read past the grace must NOT abort
    assert inp._charge_step(0, 0.5, True, 40, 0.3, 1.5) == "hold"


def test_charge_step_holds_while_charging_below_target():
    assert inp._charge_step(20, 0.2, True, 40, 0.3, 1.5) == "hold"


def test_charge_step_target_priority_over_max_hold():
    # both conditions true -> still 'release' (the cast happens either way)
    assert inp._charge_step(40, 1.5, True, 40, 0.3, 1.5) == "release"


# --- charge_and_release (IO) ----------------------------------------------

class _FakePyAutoGUI:
    FAILSAFE = False  # check_failsafe early-returns; no position() needed

    def __init__(self):
        self.calls = []

    def moveTo(self, x, y):
        self.calls.append(("moveTo", x, y))

    def mouseDown(self):
        self.calls.append(("mouseDown",))

    def mouseUp(self):
        self.calls.append(("mouseUp",))


class _Clock:
    """A fake monotonic clock advanced only by sleep() — deterministic timing."""
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _patch(monkeypatch, fake, clock):
    monkeypatch.setattr(inp, "pyautogui", fake)
    monkeypatch.setattr(inp.time, "time", clock.time)
    monkeypatch.setattr(inp.time, "sleep", clock.sleep)


def test_charge_release_fires_at_target(monkeypatch):
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)
    reads = iter([0, 8, 25, 44, 60])

    charge, ready = inp.charge_and_release(
        100, 200, target_charge=40, read_charge=lambda: next(reads),
        poll_s=0.025, ready_grace_s=0.3, max_hold_s=1.5, jitter=0)

    assert ready is True
    assert charge == 44                       # first poll >= 40
    seq = [c[0] for c in fake.calls]
    assert seq[0] == "moveTo" and seq[1] == "mouseDown"
    assert seq[-1] == "mouseUp"               # released via finally


def test_charge_release_aborts_when_rod_not_ready(monkeypatch):
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)

    charge, ready = inp.charge_and_release(
        0, 0, target_charge=40, read_charge=lambda: 0,   # never charges
        poll_s=0.025, ready_grace_s=0.3, max_hold_s=1.5, jitter=0)

    assert ready is False
    assert charge == 0
    assert ("mouseUp",) in fake.calls                    # button released


def test_charge_release_caps_at_max_hold_when_target_unreachable(monkeypatch):
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)
    # bar saturates at 60 but we asked for 99 -> hold to max, release at 60
    charge, ready = inp.charge_and_release(
        0, 0, target_charge=99, read_charge=lambda: 60,
        poll_s=0.025, ready_grace_s=0.3, max_hold_s=1.0, jitter=0)

    assert ready is True
    assert charge == 60
    assert clock.t >= 1.0                                 # held to the cap


def test_charge_release_releases_button_on_exception(monkeypatch):
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)

    def boom():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        inp.charge_and_release(0, 0, 40, boom, jitter=0)
    assert ("mouseUp",) in fake.calls
