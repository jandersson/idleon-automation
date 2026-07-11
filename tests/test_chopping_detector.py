"""Tests for chopping detector — leftmost-column and analyze_bar zone lookup."""
import numpy as np

from minigames.chopping.detector import (
    _leftmost_column,
    analyze_bar,
    bar_pixel_count,
    eased_time_to_red_ms,
    gold_distance_ahead,
    infer_vmax,
    leaf_vx_px_s,
    red_distance_ahead,
    zone_layout,
)


def test_leftmost_column_finds_first_qualifying_column():
    mask = np.zeros((20, 100), dtype=np.uint8)
    # Place a "leaf" with 5 vertical pixels at x=30..32.
    mask[5:10, 30:33] = 255
    # And isolated noise at x=10 (only 1 pixel).
    mask[5, 10] = 255

    # min_pixels_per_col = 2 should skip the noise and find x=30.
    assert _leftmost_column(mask, min_pixels_per_col=2) == 30


def test_leftmost_column_returns_none_on_empty_mask():
    mask = np.zeros((20, 100), dtype=np.uint8)
    assert _leftmost_column(mask, min_pixels_per_col=2) is None


def test_leftmost_column_returns_none_when_all_below_threshold():
    mask = np.zeros((20, 100), dtype=np.uint8)
    # Every column has 1 pixel — below the default 2-pixel threshold.
    mask[0, :] = 255
    assert _leftmost_column(mask, min_pixels_per_col=2) is None


def _bar_with_zones(green=(20, 60), gold=(60, 80), red_low=(0, 20), red_high=(80, 100)):
    """Make a synthetic 100x20 bar with colored zones in BGRA."""
    bgra = np.zeros((20, 100, 4), dtype=np.uint8)
    bgra[..., 3] = 255
    # BGR colors: green=(0,255,0), gold/yellow=(0,200,255), red=(0,0,255)
    bgra[:, green[0]:green[1], 1] = 255  # green
    bgra[:, gold[0]:gold[1], 1] = 200
    bgra[:, gold[0]:gold[1], 2] = 255  # gold ≈ yellow
    bgra[:, red_low[0]:red_low[1], 2] = 255  # red
    bgra[:, red_high[0]:red_high[1], 2] = 255  # red
    return bgra


def _leaf_at(x_start: int, width: int = 4):
    """A 30-tall leaf strip; bright green leaf body in BGRA."""
    leaf = np.zeros((30, 100, 4), dtype=np.uint8)
    leaf[..., 3] = 255
    # Leaf is bright green (matching LEAF_HSV: H 30-80, S/V 60+)
    leaf[5:25, x_start:x_start + width, 1] = 220
    return leaf


def test_analyze_bar_returns_green_for_leaf_over_green_zone():
    bar = _bar_with_zones()
    leaf = _leaf_at(x_start=40)  # 40 is inside the green zone (20..60)

    leaf_x, zone = analyze_bar(bar, leaf_frame=leaf)
    assert leaf_x == 40
    assert zone == "green"


def test_zone_layout_run_length_encodes_the_bar():
    # _bar_with_zones: red 0-20, green 20-60, gold 60-80, red 80-100.
    bar = _bar_with_zones()
    assert zone_layout(bar) == "r20 g40 o20 r20"


def test_zone_layout_marks_uncolored_columns_none():
    # Leave a gap: green 20-60 only, rest of the bar black.
    bar = _bar_with_zones(green=(20, 60), gold=(0, 0), red_low=(0, 0), red_high=(0, 0))
    assert zone_layout(bar) == "n20 g40 n40"


def test_red_distance_ahead_respects_direction():
    # red 0-20 and 80-100, leaf at x=64.
    bar = _bar_with_zones()
    # Moving right: nearest red ahead starts at 80 -> 16px.
    assert red_distance_ahead(bar, 64, direction=+1) == 16
    # Moving left: nearest red ahead is the left block's right edge (19) -> 45px.
    assert red_distance_ahead(bar, 64, direction=-1) == 45


def test_gold_distance_ahead_respects_direction():
    # gold 60-80; leaf at x=40.
    bar = _bar_with_zones()
    assert gold_distance_ahead(bar, 40, direction=+1) == 20
    assert gold_distance_ahead(bar, 40, direction=-1) is None
    # Standing inside gold counts as 0 ahead.
    assert gold_distance_ahead(bar, 70, direction=+1) == 0


def test_red_distance_ahead_none_when_no_red_in_direction():
    # Red only on the left (0-20); leaf at 50 moving right has none ahead.
    bar = _bar_with_zones(red_low=(0, 20), red_high=(0, 0))
    assert red_distance_ahead(bar, 50, direction=+1) is None
    assert red_distance_ahead(bar, 50, direction=-1) == 31


def test_leaf_vx_uses_oldest_sample_inside_window():
    # 100 px over 0.1s -> +1000 px/s, regardless of intermediate samples.
    track = [(0.0, 0), (0.05, 50), (0.1, 100)]
    assert leaf_vx_px_s(track, min_span_s=0.02, max_span_s=0.2) == 1000.0


def test_leaf_vx_none_until_min_span():
    assert leaf_vx_px_s([]) is None
    assert leaf_vx_px_s([(0.0, 10)]) is None
    # Two samples 5ms apart — below the 20ms minimum span.
    assert leaf_vx_px_s([(0.0, 10), (0.005, 12)]) is None


def test_leaf_vx_ignores_samples_older_than_max_span():
    # The 1.0s-old sample (pre-cooldown, possibly pre-bounce) must not count.
    track = [(0.0, 200), (1.0, 10), (1.05, 25)]
    assert abs(leaf_vx_px_s(track, min_span_s=0.02, max_span_s=0.2) - 300.0) < 1e-9


def test_leaf_vx_only_uses_samples_since_direction_reversal():
    # Leaf swept left to x=20, bounced, now moving right. Samples from
    # before the bounce must not dilute the estimate: since the
    # reversal it covered 80px in 0.1s -> +800 px/s. (A naive
    # oldest-in-window estimate over the full track would give 0.)
    track = [(0.0, 100), (0.05, 60), (0.1, 20), (0.15, 60), (0.2, 100)]
    assert abs(leaf_vx_px_s(track, min_span_s=0.02, max_span_s=0.2) - 800.0) < 1e-9


def test_leaf_vx_none_right_after_reversal():
    # Only one post-reversal pair, 10ms apart — below min span, and the
    # pre-reversal samples must not be used to satisfy it.
    track = [(0.0, 100), (0.05, 60), (0.1, 20), (0.11, 24)]
    assert leaf_vx_px_s(track, min_span_s=0.02, max_span_s=0.2) is None


def test_leaf_vx_zero_dx_samples_do_not_break_the_run():
    # A flat pair mid-run (sub-pixel motion) keeps the run intact.
    track = [(0.0, 10), (0.05, 10), (0.1, 40)]
    assert abs(leaf_vx_px_s(track, min_span_s=0.02, max_span_s=0.2) - 300.0) < 1e-9


def test_infer_vmax_is_jitter_robust():
    # Mid-bar samples at ~true speed 500 (sin(theta)=1 at x=W/2), plus
    # one absurd 1800 px/s jitter spike — the median must ignore it.
    w = 222
    samples = [(111, 500.0), (111, 510.0), (111, 490.0), (111, 505.0), (111, 1800.0)]
    v = infer_vmax(samples, w)
    assert 490 <= v <= 510


def test_infer_vmax_corrects_edge_samples_upward():
    # An edge-region sample moves slower than V_max; the sin correction
    # recovers the peak. x=31 on a 222 bar: 1-2x/W=0.72, sin=0.69.
    w = 222
    samples = [(31, 345.0)] * 5
    v = infer_vmax(samples, w)
    assert 480 <= v <= 520  # 345 / 0.69 ~= 500


def test_infer_vmax_needs_five_samples():
    assert infer_vmax([(111, 500.0)] * 4, 222) is None


def test_eased_time_to_red_position_awareness():
    # Same 60px to red, same V_max: launched from the slow edge region
    # takes LONGER than the linear estimate; launched mid-bar toward a
    # nearby red is FASTER than naive distance/instantaneous-vx says.
    w, v_max = 222, 600
    edge = eased_time_to_red_ms(20, +1, 60, v_max, w)
    mid = eased_time_to_red_ms(111, +1, 60, v_max, w)
    assert edge > mid
    # Mid-bar at peak speed: time ~= d / v_max as the lower bound.
    assert mid >= int(60 / v_max * 1000) - 1


def test_eased_time_to_red_mirrors_direction():
    w, v_max = 222, 600
    rightward = eased_time_to_red_ms(60, +1, 50, v_max, w)
    leftward = eased_time_to_red_ms(222 - 60, -1, 50, v_max, w)
    assert rightward == leftward


def test_eased_time_to_red_validated_death_cases():
    # The two computable recorded deaths must price under the 120ms
    # budget (scripts/validate_chop_gate.py is the full back-test):
    # 00:13 chop 3 (x=152 rightward, 19px ahead, V_max ~750) and
    # 01:08 chop 20 (x=115 rightward, 64px ahead, V_max ~626).
    assert eased_time_to_red_ms(152, +1, 19, 750, 222) < 120
    assert eased_time_to_red_ms(115, +1, 64, 626, 222) < 120
    # A green-entry survival from the same data prices comfortably over:
    # 01:39 chop 2 (x=161 leftward, 106px ahead, V_max ~430).
    assert eased_time_to_red_ms(161, -1, 106, 430, 222) > 200


def test_analyze_bar_returns_gold_for_leaf_over_gold_zone():
    bar = _bar_with_zones()
    leaf = _leaf_at(x_start=65)  # gold zone is 60..80

    _, zone = analyze_bar(bar, leaf_frame=leaf)
    assert zone == "gold"


def test_analyze_bar_returns_red_for_leaf_over_red_zone():
    bar = _bar_with_zones()
    leaf = _leaf_at(x_start=5)  # red_low is 0..20

    _, zone = analyze_bar(bar, leaf_frame=leaf)
    assert zone == "red"


def test_analyze_bar_returns_none_when_leaf_absent():
    bar = _bar_with_zones()
    leaf = np.zeros((30, 100, 4), dtype=np.uint8)
    leaf[..., 3] = 255  # all-black leaf strip — no leaf present

    leaf_x, zone = analyze_bar(bar, leaf_frame=leaf)
    assert leaf_x is None
    assert zone == "none"


def test_bar_pixel_count_high_when_zones_present():
    bar = _bar_with_zones()
    # Bar is 20 rows × 100 cols, fully colored — expect thousands of pixels.
    assert bar_pixel_count(bar) > 500


def test_bar_pixel_count_near_zero_when_bar_blank():
    # All-neutral (black) frame — no green/red/gold should match.
    blank = np.zeros((20, 100, 4), dtype=np.uint8)
    blank[..., 3] = 255
    assert bar_pixel_count(blank) < 30


def test_gold_window_ms_none_without_gold():
    from minigames.chopping.detector import gold_window_ms
    bar = _bar_with_zones(gold=(60, 60))  # zero-width gold — none on the bar
    assert gold_window_ms(bar, v_max=500.0) is None


def test_gold_window_ms_prices_red_sandwich_low():
    from minigames.chopping.detector import gold_window_ms
    # The 21:01 run's doomed shape, scaled to the 100px test bar:
    # green | red | gold | red — both gold entries have red hard ahead.
    bar = _bar_with_zones(green=(10, 55), red_low=(55, 62),
                          gold=(62, 78), red_high=(78, 90))
    tight = gold_window_ms(bar, v_max=600.0)
    # 16px of gold at 600 px/s mid-bar can't come close to the 120ms gate.
    assert tight is not None and tight < 80


def test_gold_window_ms_prices_open_runway_high():
    from minigames.chopping.detector import gold_window_ms
    # Gold with a long green runway before the far red: entering from
    # the left, red is ~55px past the gold entry.
    bar = _bar_with_zones(green=(20, 60), gold=(60, 75),
                          red_low=(0, 10), red_high=(90, 100))
    open_w = gold_window_ms(bar, v_max=400.0)
    assert open_w is not None and open_w > 100


def test_gold_window_ms_unbounded_without_red_past_gold():
    from minigames.chopping.detector import gold_window_ms
    # No red at all in one travel direction past the gold.
    bar = _bar_with_zones(green=(20, 60), gold=(60, 80),
                          red_low=(0, 10), red_high=(95, 95))
    assert gold_window_ms(bar, v_max=800.0) == 10**6


def test_gold_window_ms_slower_leaf_prices_higher():
    from minigames.chopping.detector import gold_window_ms
    bar = _bar_with_zones(green=(10, 55), red_low=(55, 62),
                          gold=(62, 78), red_high=(78, 90))
    slow = gold_window_ms(bar, v_max=250.0)
    fast = gold_window_ms(bar, v_max=700.0)
    assert slow > fast


# --- overlay auto-location (find_bar_rect / derive_overlay_regions) ---

def _window_with_bar(bar=(300, 200, 220, 14), caps=True):
    """Synthetic 960x572 BGRA window: the minigame bar (red|green|red)
    plus the usual color noise — a green-only XP bar, a red-only HP
    bar, a small green button blob right of the bar."""
    win = np.zeros((572, 960, 4), dtype=np.uint8)
    win[..., 3] = 255
    x, y, w, h = bar
    win[y:y+h, x:x+50, 2] = 255                      # red left
    win[y:y+h, x+50:x+170, 1] = 255                  # green middle
    win[y:y+h, x+170:x+w, 2] = 255                   # red right
    if caps:  # decorative green end caps 2px off the playfield
        win[y-2:y+h+2, x-9:x-3, 1] = 200
        win[y-2:y+h+2, x+w+3:x+w+9, 1] = 200
    win[540:548, 100:420, 1] = 255                   # XP bar: green only
    win[515:524, 100:400, 2] = 255                   # HP bar: red only
    win[y:y+22, x+w+14:x+w+58, 1] = 220              # Chop button blob
    return win


def test_find_bar_rect_locates_the_playfield_not_the_caps():
    from minigames.chopping.detector import find_bar_rect
    win = _window_with_bar()
    rect = find_bar_rect(win)
    assert rect is not None
    left, top, w, h = rect
    assert abs(left - 300) <= 1 and abs(top - 200) <= 2
    assert abs(w - 220) <= 2 and 10 <= h <= 20


def test_find_bar_rect_ignores_single_color_bars():
    from minigames.chopping.detector import find_bar_rect
    win = _window_with_bar()
    # Remove the real bar: only XP/HP/button noise remains.
    win[198:216, 280:540] = 0
    win[..., 3] = 255
    assert find_bar_rect(win) is None


def test_derive_overlay_regions_offsets():
    from minigames.chopping.detector import derive_overlay_regions
    regs = derive_overlay_regions((300, 200, 222, 18))
    assert regs["bar"] == {"left": 300, "top": 200, "width": 222, "height": 18}
    assert regs["leaf"]["top"] == 200 - regs["leaf"]["height"]
    assert regs["leaf"]["left"] == 300 and regs["leaf"]["width"] == 222
    assert regs["button"]["left"] == 300 + 234   # bar right + 12
    assert regs["score"]["top"] == 200 + 18 + 1
    assert regs["score"]["left"] == 298


def test_leaf_left_edge_prefers_the_widest_run():
    from minigames.chopping.detector import _leaf_left_edge
    mask = np.zeros((20, 100), dtype=np.uint8)
    mask[5:8, 2:4] = 255      # narrow intruder (an end cap's spill)
    mask[3:18, 60:71] = 255   # the leaf body
    assert _leaf_left_edge(mask) == 60


def test_leaf_left_edge_none_on_empty():
    from minigames.chopping.detector import _leaf_left_edge
    assert _leaf_left_edge(np.zeros((20, 100), dtype=np.uint8)) is None


def test_find_play_button_matches_pasted_template():
    import cv2
    import minigames.chopping.detector as D
    tpl = cv2.imread(str(D.ASSETS / "play_button.png"), cv2.IMREAD_COLOR)
    assert tpl is not None, "play_button.png template missing"
    win = np.zeros((572, 960, 4), dtype=np.uint8)
    win[..., 3] = 255
    th, tw = tpl.shape[:2]
    win[150:150+th, 480:480+tw, :3] = tpl
    D._play_button_template = None  # reset the lazy cache
    center = D.find_play_button(win)
    assert center is not None
    cx, cy = center
    assert abs(cx - (480 + tw // 2)) <= 2 and abs(cy - (150 + th // 2)) <= 2


def test_find_play_button_none_on_blank_frame():
    import minigames.chopping.detector as D
    win = np.zeros((572, 960, 4), dtype=np.uint8)
    win[..., 3] = 255
    D._play_button_template = None
    assert D.find_play_button(win) is None
