"""Tests for scripts.bootstrap_digit_templates — the launcher 'Refresh from DB' path."""
from scripts.bootstrap_digit_templates import GAMES, bootstrap_game


def test_unsupported_game_returns_empty_not_keyerror():
    """Mining (and any game not in GAMES) bootstraps digit templates offline
    from labelled digit_capture crops, not from the DB — so bootstrap_game must
    return {} cleanly rather than KeyError. Regression for the launcher
    'Refresh from DB' button raising KeyError('mining')."""
    assert "darts" in GAMES and "hoops" in GAMES   # the DB-refresh games
    assert "mining" not in GAMES                    # mining isn't one
    assert bootstrap_game("mining") == {}           # graceful, no KeyError
    assert bootstrap_game("does-not-exist") == {}
