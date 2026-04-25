import cv2
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def find_target(frame: np.ndarray) -> tuple[int, int] | None:
    """Locate whatever the bot needs to aim at / time off of in the dart UI.

    Returns center (x, y) within the cropped board region, or None if not found.

    TODO: implement once we know the darts mechanics. Likely candidates:
    - template-match a moving aim cursor and fire when it overlaps the bullseye
    - HSV-mask a colored sweep / power bar and fire at the right phase
    """
    _ = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # placeholder — keep BGRA→BGR convention
    return None
