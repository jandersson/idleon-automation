"""Tests for common.input closed-loop charge-and-release (fishing cast).

`_charge_step` is the pure poll decision (tested exhaustively); the IO wrapper
`charge_and_release` is exercised with a fake pyautogui + a controllable clock,
mirroring tests/test_input_hold.py's fake-pyautogui style.
"""
import itertools

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


# --- _charge_step release-lead (the "lead cast", overshoot fix) -------------

def test_charge_step_rate_zero_is_old_behaviour():
    # default charge_rate=0 -> no lead -> releases exactly at target (back-compat)
    assert inp._charge_step(39, 0.2, True, 40, 0.3, 1.5) == "hold"
    assert inp._charge_step(40, 0.2, True, 40, 0.3, 1.5) == "release"


def test_charge_step_leads_release_when_rising_fast():
    # rising 200/s, lead = 200*0.05 = 10 -> release when charge+10 >= target,
    # i.e. ~10 below target (cancels the latency that else overshoots).
    assert inp._charge_step(29, 0.2, True, 40, 0.3, 1.5, charge_rate=200) == "hold"   # 39 < 40
    assert inp._charge_step(30, 0.2, True, 40, 0.3, 1.5, charge_rate=200) == "release" # 40 >= 40


def test_charge_step_lead_is_capped_against_a_rate_spike():
    # a noisy huge rate can't release wildly early: lead is capped at 12.
    assert inp._charge_step(27, 0.2, True, 40, 0.3, 1.5, charge_rate=99999) == "hold"   # 27+12=39<40
    assert inp._charge_step(28, 0.2, True, 40, 0.3, 1.5, charge_rate=99999) == "release" # 28+12=40>=40


def test_charge_step_negative_rate_clamped_no_early_release():
    # a downward blip (fill drops) must not project a release; clamp rate>=0.
    assert inp._charge_step(35, 0.2, True, 40, 0.3, 1.5, charge_rate=-500) == "hold"


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


def test_charge_release_leads_a_fast_rise(monkeypatch):
    # A fast rise (5/poll @ poll_s=0.025 -> 200/s) RELEASES EARLY: the release-
    # lead (200*0.05=10) fires when the fill is projected to reach target, ~10
    # below it, so the cast lands near target instead of overshooting (#eel-miss).
    # The bare reading at release is ~30, but the RETURN is the projected cast
    # power (reading + lead) == 40, the model-consistent training feature.
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)
    consumed = []
    ramp = itertools.chain([0, 5, 10, 15, 20, 25, 30, 35, 40, 45], itertools.repeat(60))

    def rd():
        v = next(ramp)
        consumed.append(v)
        return v

    charge, ready = inp.charge_and_release(
        100, 200, target_charge=40, read_charge=rd,
        poll_s=0.025, ready_grace_s=0.3, max_hold_s=1.5, jitter=0)

    assert ready is True
    assert max(consumed) == 30                  # released at reading 30 — before target (lead)
    assert charge == 40                         # ...but returns projected power = reading + lead
    seq = [c[0] for c in fake.calls]
    assert seq[0] == "moveTo" and seq[1] == "mouseDown"
    assert seq[-1] == "mouseUp"                 # released via finally


def test_charge_release_slow_rise_minimal_lead(monkeypatch):
    # A slow rise (3/poll @ poll_s=0.1 -> 30/s) gets only a tiny lead
    # (30*0.05=1.5), so the common, already-accurate cast releases ~at target —
    # the lead can't overcorrect a slow cast into an undershoot.
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)
    consumed = []
    ramp = itertools.chain([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
                           itertools.repeat(60))

    def rd():
        v = next(ramp)
        consumed.append(v)
        return v

    charge, ready = inp.charge_and_release(
        100, 200, target_charge=40, read_charge=rd,
        poll_s=0.1, ready_grace_s=0.3, max_hold_s=5.0, jitter=0)

    assert ready is True
    assert max(consumed) == 39                  # released at reading 39 — barely early
    assert charge == 40                         # projected 39 + 1.5 -> 40


def test_charge_release_suppresses_abort_after_charging_then_transient_zero(monkeypatch):
    # Loop-level stickiness: once the fill crosses the floor the rod is
    # 'charging', so a later transient 0 (a noisy find_charge_level read) past
    # the grace window must NOT abort. Guards against a non-sticky regression
    # (seen_charging = charge >= floor), which ships green without this.
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)
    # poll_s 0.2 so the transient 0 (read #3) lands at t=0.4 > grace 0.3
    reads = itertools.chain([0, 20, 0, 35, 44], itertools.repeat(44))

    charge, ready = inp.charge_and_release(
        0, 0, target_charge=40, read_charge=lambda: next(reads),
        poll_s=0.2, ready_grace_s=0.3, max_hold_s=1.5, jitter=0)

    # released at reading 44 (already >= target); return is projected 44 + lead(~2)
    assert ready is True and charge == 46


def test_charge_release_speckle_below_floor_still_aborts(monkeypatch):
    # A lone 1px red speckle (below ready_floor) — e.g. the reeling bobber
    # grazing the x<10 strip — must NOT latch 'charging': the not-ready abort
    # still fires. With the old `charge > 0` latch this cast would be burned.
    fake, clock = _FakePyAutoGUI(), _Clock()
    _patch(monkeypatch, fake, clock)
    reads = itertools.chain([0, 1, 0], itertools.repeat(0))

    charge, ready = inp.charge_and_release(
        0, 0, target_charge=40, read_charge=lambda: next(reads),
        poll_s=0.025, ready_grace_s=0.3, max_hold_s=1.5, ready_floor=2, jitter=0)

    assert ready is False and charge == 0


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
