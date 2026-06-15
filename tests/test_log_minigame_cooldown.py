"""Tests for scripts/log_minigame_cooldown.py — the cooldown-decay back-out
(the pure piece; the save read is live-only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.log_minigame_cooldown import reconstruct_set_ticks


def test_no_age_is_identity():
    assert reconstruct_set_ticks(153, 0.0) == 153


def test_backs_out_decay_at_5_ticks_per_second():
    # read 150t, 2.0s after the flush -> was 150 + 2*5 = 160t at the flush
    assert reconstruct_set_ticks(150, 2.0) == 160
    assert reconstruct_set_ticks(100, 1.0) == 105  # default 5 ticks/sec


def test_rounds_to_nearest_tick():
    assert reconstruct_set_ticks(100.4, 0.0) == 100
    assert reconstruct_set_ticks(100, 0.14) == 101  # 100 + 0.7 -> 101


def test_custom_ticks_per_second():
    assert reconstruct_set_ticks(100, 2.0, ticks_per_second=10) == 120
