"""Pure jump-policy logic for the mining bot — no IO, no CV.

Split out from main.py so the fire decision is unit-testable in isolation
(the loop just samples the detector and asks should_jump). The load-bearing
piece is the airborne guard: in the cart minigame a click while the cart is
GROUNDED is a jump, but a click while it's AIRBORNE is a *slam* (a fast
descent). The cooldown (JUMP_COOLDOWN_S) only blanks the first ~0.6s after a
jump, but the arc stays airborne ~0.9s — so a second pit entering the
trigger window in that ~0.3s tail used to pass the predicate and fire a
click the game read as a slam, driving the cart down into the approaching
pit. That is the mechanism behind the documented "spaced-obstacle" death
(see docs/mining_plan.md). should_jump refuses to fire while airborne.

Altitude is read from cart_y against a grounded BASELINE rather than from
(plank_y - cart_y): plank_y is rock-stable while grounded but jitters a few
px during the jump (exactly when the difference would be near the launch/
land boundary), whereas the grounded cart_y is extremely stable (193 for
every grounded frame of botrun_20260615_012340) and the peak is far away
(~133), so a baseline-relative test has a wide, noise-proof margin.
"""
from __future__ import annotations

from collections import deque
from typing import Optional


# A cart this many px ABOVE its grounded resting y counts as airborne.
# Screen y grows downward, so a raised (airborne) cart has a SMALLER y. The
# grounded->peak swing is ~60px (193->133) so a generous 18px threshold
# cleanly brackets the resting jitter while still catching the whole up-arc.
JUMP_GROUNDED_EPS_PX = 18


class GroundedBaseline:
    """Rolling estimate of the cart's resting (grounded) screen-y.

    The cart sits at its MAX y when grounded and rises (smaller y) on a jump;
    a death pushes it slightly below the plank (a bit larger y) just before
    it vanishes. So the grounded level is the high-y end of a recent window,
    and the MAX over the window tracks it robustly: airborne frames (small y)
    can only ever pull a mean/median down (risking a false 'airborne' verdict
    on a grounded cart — the dangerous direction), but they never raise the
    max. The occasional death frame inflates the max a few px, which only
    makes the guard *less* eager to suppress — the safe direction.

    baseline() returns None until `warmup` samples have been seen, so the
    airborne guard defaults to 'grounded' (never suppresses a survival jump)
    during the first frames of a run before the resting level is known.
    """

    def __init__(self, window: int = 45, warmup: int = 8):
        self._buf: deque[int] = deque(maxlen=window)
        self._warmup = warmup

    def update(self, cart_y: Optional[int]) -> None:
        """Feed this frame's cart_y (None when the cart isn't detected —
        ignored, so a lost cart doesn't corrupt the resting estimate)."""
        if cart_y is not None:
            self._buf.append(int(cart_y))

    def baseline(self) -> Optional[int]:
        if len(self._buf) < self._warmup:
            return None
        return max(self._buf)


def is_cart_airborne(cart_y: Optional[int], grounded_baseline_y: Optional[int],
                     eps: int = JUMP_GROUNDED_EPS_PX) -> bool:
    """True when the cart is clearly above its grounded resting level.

    Unknown inputs (no cart, no warmed-up baseline) return False — i.e. treat
    as grounded — so the airborne guard never suppresses a needed survival
    jump just because altitude is momentarily unknown. The conservative
    default is 'fire if in doubt'."""
    if cart_y is None or grounded_baseline_y is None:
        return False
    return cart_y < grounded_baseline_y - eps


def should_jump(*, cart, plank_y, terrain, now: float, last_click_time: float,
                grounded_baseline_y: Optional[int], cooldown_s: float,
                trig_min: int, trig_max: int,
                airborne_eps: int = JUMP_GROUNDED_EPS_PX) -> bool:
    """The auto-mode fire predicate. Returns True iff a jump click should
    fire this frame.

    Gates, in cheap-to-expensive order: cart + plank present; nearest terrain
    is a pit; its distance is inside [trig_min, trig_max]; the post-jump
    cooldown has expired; and the cart is NOT airborne (see module docstring —
    an airborne click is a lethal slam). Click POSITION is irrelevant (Idleon
    reads a click as a button press), so this returns only the yes/no — the
    caller picks where to click."""
    if cart is None or plank_y is None or terrain is None:
        return False
    if terrain.get("kind") != "pit":
        return False
    dist = terrain.get("distance_px")
    if dist is None or not (trig_min <= dist <= trig_max):
        return False
    if now - last_click_time < cooldown_s:
        return False
    if is_cart_airborne(cart[1], grounded_baseline_y, airborne_eps):
        return False
    return True
