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


def _ocr_one(binary: np.ndarray, psm: int) -> int | None:
    """Single tesseract call. Returns parsed int or None."""
    global _BINARY_WARNED
    try:
        text = pytesseract.image_to_string(
            binary,
            config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
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


def read_score(crop_gray: np.ndarray, scale: int = 6) -> int | None:
    """Return the integer in the crop, or None if OCR isn't available, the
    crop is too small, or the OCR passes don't agree.

    Multi-pass voting: we run tesseract with several preprocessings/PSM
    modes and only trust the result if at least 2 passes agree. The Idleon
    score font is small and pixel-art (not what tesseract was trained on),
    so individual passes occasionally misread "0" as "1". Cross-mode
    agreement filters those out — if a digit is genuinely 0, multiple
    different preprocessings will all see 0; if tesseract is hallucinating,
    different preprocessings will produce different hallucinations.
    """
    if not _HAVE_PYTESSERACT or crop_gray is None or crop_gray.size == 0:
        return None
    h, w = crop_gray.shape[:2]
    if h < 4 or w < 4:
        return None
    # Generate a few preprocessings to OCR independently.
    upscaled = cv2.resize(crop_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    # Slightly dilated copy so digit strokes are thicker (helps tesseract on
    # very thin pixel fonts).
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    # PSM 8 = single word, PSM 10 = single character. Different segmentation
    # assumptions; agreement across both is a stronger signal.
    candidates = [
        _ocr_one(binary, 8),
        _ocr_one(binary, 10),
        _ocr_one(inverted, 8),
        _ocr_one(dilated, 8),
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    # Take the most common; require at least 2 votes for agreement.
    from collections import Counter
    most_common, count = Counter(candidates).most_common(1)[0]
    if count < 2:
        return None
    return most_common
