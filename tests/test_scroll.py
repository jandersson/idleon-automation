"""Tests for the launcher mouse-wheel helper (#93) — pure delta math only.

The binding/routing needs a live Tk display, so it stays manual; the unit
conversion is the one piece of logic worth pinning."""
from ui.launcher.scroll import _wheel_units


def test_wheel_units_maps_delta_to_scroll_direction():
    assert _wheel_units(120) == -1   # wheel up   → scroll up
    assert _wheel_units(-120) == 1   # wheel down → scroll down


def test_wheel_units_scales_with_notches():
    assert _wheel_units(240) == -2
    assert _wheel_units(-360) == 3


def test_wheel_units_rounds_sub_notch_to_zero():
    # High-res wheels send <120 per event; round toward 0 (no jump).
    assert _wheel_units(60) == 0
    assert _wheel_units(-60) == 0
