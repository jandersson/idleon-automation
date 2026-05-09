"""Mining detector — stub.

Two functions to fill in once real frames are captured:

- find_cart(frame) -> (x, y) | None
    Locate the mining-cart's reference point (likely the wheels or the
    cart's leading edge). Cart's X is roughly fixed on screen; mostly we
    need its Y to know if it's grounded vs airborne.

- find_next_terrain(frame, cart) -> dict | None
    Look ahead of the cart and classify the next terrain feature:
    {"kind": "ore" | "pit" | "trap", "x": int, "distance_px": int}.
    Probably HSV masking — ore should have a distinct sprite color, pits
    are dark voids in the track.
"""
from typing import Optional, Tuple


def find_cart(frame) -> Optional[Tuple[int, int]]:
    return None


def find_next_terrain(frame, cart) -> Optional[dict]:
    return None
