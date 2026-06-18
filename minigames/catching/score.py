"""Catching PTS score reader.

The tolerant pixel-digit reader now lives in ``common.score_digits`` (shared
with the fishing bot); this module re-exports it so catching's callers and
tests keep importing ``minigames.catching.score``. Catching's font-specific
digit templates stay under ``assets/digit_templates`` and are bootstrapped /
grown with ``catching-capture-digits``. See ``common/score_digits.py`` for the
algorithm (Otsu-minority binarize -> digit-height components -> tolerant
zero-mean correlation match, stopping at the first non-digit to drop "PTS").
"""
from common.score_digits import (
    DIGIT_CANON_HW,
    DIGIT_HEIGHT_FRAC,
    DIGIT_MATCH_THRESHOLD,
    DIGIT_MAX_WIDTH,
    DIGIT_MIN_AREA,
    DIGIT_MIN_HEIGHT,
    DIGIT_MIN_WIDTH,
    HLINE_ROW_FILL,
    WHITE_FILL_S_MAX,
    WHITE_FILL_V_MIN,
    _binarize_glyph,
    _canon,
    _corr,
    _default_binarize,
    _strip_horizontal_lines,
    _to_gray,
    binarize_white_fill,
    digit_components,
    load_digit_templates,
    make_pts_reader,
    match_digit,
    parse_pts,
    read_pts_from_crop,
    save_digit_template,
)

__all__ = [
    "DIGIT_CANON_HW", "DIGIT_HEIGHT_FRAC", "DIGIT_MATCH_THRESHOLD",
    "DIGIT_MAX_WIDTH", "DIGIT_MIN_AREA", "DIGIT_MIN_HEIGHT", "DIGIT_MIN_WIDTH",
    "HLINE_ROW_FILL", "WHITE_FILL_S_MAX", "WHITE_FILL_V_MIN", "_binarize_glyph",
    "_canon", "_corr", "_default_binarize", "_strip_horizontal_lines",
    "_to_gray", "binarize_white_fill", "digit_components",
    "load_digit_templates", "make_pts_reader", "match_digit", "parse_pts",
    "read_pts_from_crop", "save_digit_template",
]
