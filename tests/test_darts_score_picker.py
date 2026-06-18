"""Tests for the multi-pass post-throw score picker in darts.main."""
from minigames.darts.main import _pick_best_score_after, VALID_SCORE_INCREMENTS


def test_valid_increments_match_stripe_color_keys():
    """Sanity: the valid-increment set is the same as the stripe-color mapping."""
    from minigames.darts.main import STRIPE_COLOR_BY_INCREMENT
    assert VALID_SCORE_INCREMENTS == frozenset(STRIPE_COLOR_BY_INCREMENT.keys())


def test_empty_candidates_returns_none_none():
    assert _pick_best_score_after(score_before_int=10, post_readings=[]) == (None, None)


def test_all_none_candidates_returns_none_none():
    assert _pick_best_score_after(score_before_int=10, post_readings=[None, None, None]) == (None, None)


def test_no_score_before_returns_last_non_none_with_no_increment():
    """If we have no pre-throw baseline we can't compute increments,
    so return the most recent reading we managed to OCR."""
    after, incr = _pick_best_score_after(score_before_int=None, post_readings=[None, 42, None])
    assert after == 42
    assert incr is None


def test_single_valid_increment_chosen():
    """One reading that produces a valid increment (+5 bullseye) wins."""
    # score_before=10, post readings include a 15 (=+5, bullseye)
    after, incr = _pick_best_score_after(
        score_before_int=10,
        post_readings=[15],
    )
    assert (after, incr) == (15, 5)


def test_invalid_reading_filtered_in_favor_of_valid():
    """195 (the canonical tesseract bullseye misread) is rejected when a
    valid +5 reading is also present."""
    after, incr = _pick_best_score_after(
        score_before_int=10,
        post_readings=[195, 15, None, -4],  # 15 is the only valid +5
    )
    assert (after, incr) == (15, 5)


def test_vote_among_multiple_valid_increments():
    """Most-common valid increment wins when readings disagree."""
    # score_before=20. readings: 23 (=+3), 25 (=+5), 23 (=+3 again) -> +3 wins 2-1
    after, incr = _pick_best_score_after(
        score_before_int=20,
        post_readings=[23, 25, 23],
    )
    assert incr == 3
    assert after == 23


def test_no_valid_increment_returns_none_none():
    """When every candidate produces an out-of-set increment (a misread —
    the score can only move +1/+2/+3/+5 per throw), return (None, None)
    rather than logging a bogus 10/20 (#78). hit/miss is captured
    separately by the score diff, so nothing of value is lost."""
    after, incr = _pick_best_score_after(
        score_before_int=10,
        post_readings=[None, 195, -4, None, 200],  # -5/185/190 all invalid
    )
    assert (after, incr) == (None, None)


def test_pre_update_readings_filtered_naturally():
    """Early flight frames show the pre-update score (increment=0, not
    in {1,2,3,5}) and are naturally rejected by the sanity filter."""
    after, incr = _pick_best_score_after(
        score_before_int=10,
        post_readings=[10, 10, 10, 13, 13],  # pre-update reads of 10, then updated to 13 (=+3)
    )
    assert (after, incr) == (13, 3)


def test_zero_increment_not_in_valid_set():
    """+0 isn't in the valid set (would mean no score change), so a 'no
    change' reading shouldn't win even if it appears many times."""
    after, incr = _pick_best_score_after(
        score_before_int=10,
        post_readings=[10, 10, 10, 13],  # many no-change reads, one +3
    )
    assert (after, incr) == (13, 3)
