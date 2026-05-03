"""OCR a small numeric score crop using Tesseract.

Designed for in-game score readouts that are tiny (~10-20px tall). Tesseract
needs more pixels and clean contrast to read reliably, so we upscale and
threshold before handing the crop over.

Tesseract is an external binary — install on Windows with:
    winget install --id=UB-Mannheim.TesseractOCR
or download the installer from https://github.com/UB-Mannheim/tesseract/wiki

The Python wrapper (pytesseract) is a hard dependency in pyproject.toml,
but the binary itself is optional: if it's missing or fails, read_score
returns None and the bot keeps going (we just don't log score ints).

This module is intentionally minigame-agnostic — anything with a numeric
readout (hoops, darts, future minigames) can call read_score(crop_gray).
"""
import os
import shutil
import sys

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
    _HAVE_PYTESSERACT = True
except ImportError:
    pytesseract = None
    TesseractNotFoundError = Exception  # type: ignore
    _HAVE_PYTESSERACT = False

# Print the missing-binary warning at most once per process so logs aren't
# spammed for every shot in a session.
_BINARY_WARNED = False


def _find_tesseract_binary() -> str | None:
    """Locate tesseract.exe even if it isn't on PATH. winget installs to
    Program Files but doesn't always update the user's PATH; this saves the
    user from having to fiddle with system env vars."""
    # PATH first — the easy case.
    found = shutil.which("tesseract")
    if found:
        return found
    # Common Windows install locations.
    if sys.platform == "win32":
        for path in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ):
            if os.path.isfile(path):
                return path
    return None


# Configure pytesseract to use whatever tesseract binary we can find.
if _HAVE_PYTESSERACT:
    _binary = _find_tesseract_binary()
    if _binary:
        pytesseract.pytesseract.tesseract_cmd = _binary


def read_score(crop_gray: np.ndarray, scale: int = 6) -> int | None:
    """Return the integer in the crop, or None if OCR isn't available or
    couldn't confidently parse a number.

    `crop_gray` is a small (~10-20 px tall) grayscale image with the score
    digits on a contrasting background. We upscale by `scale` (cubic) and
    Otsu-binarize before passing to Tesseract — this is what makes a 12px
    digit reliably readable.
    """
    global _BINARY_WARNED
    if not _HAVE_PYTESSERACT or crop_gray is None or crop_gray.size == 0:
        return None
    h, w = crop_gray.shape[:2]
    if h < 4 or w < 4:
        return None
    # Upscale, then binarize. White-on-black or black-on-white doesn't matter
    # to Tesseract once thresholded, but contrast does.
    upscaled = cv2.resize(crop_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(
            binary,
            config="--psm 8 -c tessedit_char_whitelist=0123456789",
        )
    except TesseractNotFoundError:
        if not _BINARY_WARNED:
            print("[score_ocr] tesseract binary not found on PATH; scores not OCR'd. "
                  "Install: winget install --id=UB-Mannheim.TesseractOCR")
            _BINARY_WARNED = True
        return None
    except Exception as e:
        if not _BINARY_WARNED:
            print(f"[score_ocr] tesseract call failed (non-fatal): {e}")
            _BINARY_WARNED = True
        return None
    text = text.strip()
    if not text or not text.isdigit():
        return None
    return int(text)
