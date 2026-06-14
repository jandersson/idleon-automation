"""Capture per-digit template candidates from a live mining run.

The PTS readout is plank-relative (the overlay floats above the player's
world position), so there's no fixed regions.json rectangle the offline
`scripts/bootstrap_digit_templates.py` tool can crop the way it does for
darts/hoops — and mining.db has no saved per-jump frames to mine either.
So we capture digit templates live instead: while the bot (or the human,
in --watch mode) plays, this saves each DISTINCT score crop to
`assets/digit_capture/<stamp>/`, deduped by its binarized pixels so one
PNG lands per rendered number (0, 1, 2, ... then 10, 11, ... as the score
climbs). Only `0.png` is seeded today; a single run that scores into the
teens cycles the readout through every digit 1-9.

After the run the crops are labelled (by eye — they render as images — or
by tesseract) and the per-digit binary components fed to
`common.score_template_ocr.save_template` to fill in
`digit_templates/1.png .. 9.png`. See docs/mining_plan.md for the
labelling step.

Dedup is by the per-component bounding-box + binary-pixel signature, not
the raw crop, so anti-aliasing / dark-background flicker between readouts
doesn't spam near-duplicate PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2

from common.score_template_ocr import binarize, extract_digit_components


def crop_signature(crop) -> Optional[str]:
    """Stable key identifying a rendered score: each digit component's
    (w, h) box plus its exact binary pixels. None when the crop has no
    digit-like components (e.g. an all-dark frame between readouts).

    Two crops of the same number share a signature regardless of where
    in the box the digits sit; two different numbers never collide
    because their digit pixels differ."""
    comps = extract_digit_components(crop)
    if not comps:
        return None
    parts = []
    for patch, box in comps:
        parts.append(f"{box[2]}x{box[3]}:{patch.tobytes().hex()}")
    return "|".join(parts)


class DigitCapturer:
    """Accumulates distinct PTS score crops over a run and writes them to
    disk for offline digit-template labelling."""

    def __init__(self, out_dir: Path, max_crops: int = 300):
        self.out_dir = Path(out_dir)
        self.max_crops = max_crops
        self._seen: set[str] = set()
        self._manifest: list[dict] = []

    @property
    def count(self) -> int:
        return len(self._manifest)

    def maybe_save(self, crop, *, frame_idx: int = -1,
                   t_rel: float = -1.0) -> bool:
        """Save the crop iff it's a not-yet-seen rendered number (and we're
        under the cap). Returns True when a new crop was written."""
        if crop is None or self.count >= self.max_crops:
            return False
        sig = crop_signature(crop)
        if sig is None or sig in self._seen:
            return False
        self._seen.add(sig)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        n = self.count
        raw_path = self.out_dir / f"crop_{n:03d}.png"
        cv2.imwrite(str(raw_path), crop)
        cv2.imwrite(str(self.out_dir / f"crop_{n:03d}_bin.png"), binarize(crop))
        self._manifest.append({
            "crop": raw_path.name,
            "n_components": len(extract_digit_components(crop)),
            "frame_idx": frame_idx,
            "t_rel": round(t_rel, 3),
        })
        return True

    def flush(self) -> Optional[Path]:
        """Write manifest.json. Returns its path, or None if nothing was
        captured. Safe to call more than once."""
        if not self._manifest:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps({"crops": self._manifest}, indent=2))
        return path
