"""Offline validation harness for the masked-ZNCC fishing score reader (#82).

Builds median glyph templates+masks from the dock corpus, then compares the
masked-ZNCC reader against the current white-fill reader over every saved score
crop + the failing beach frame. Reports regressions (value->None, value->diff),
recoveries (None->value), and the true-digit ZNCC peak distribution so the
accept threshold can be tuned on the gap. Not a pytest test (it needs the live
capture corpus); a one-off measurement tool kept under scripts/.
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.score_digits import (
    binarize_white_fill, digit_components, load_digit_templates, _canon, match_digit,
    make_pts_reader,
)
from minigames.fishing.detector import find_cast_bar
from minigames.fishing.score import score_crop

# The pre-#82 white-fill reader, constructed explicitly (score._read_pts is now
# the masked-ZNCC reader).
_read_pts = make_pts_reader(
    Path(__file__).parent.parent / "minigames/fishing/assets/digit_templates",
    binarize=binarize_white_fill)

CORPUS = sorted(glob.glob(
    str(Path(__file__).parent.parent /
        "minigames/fishing/assets/captures/botrun_2026061*/score_c*.png")))
BEACH = r"C:\Program Files (x86)\Steam\userdata\153875\760\remote\1476970\screenshots\20260618155212_1.jpg"


def _to_gray(crop):
    if crop.ndim == 2:
        return crop
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def collect_instances():
    """Label leading digits with the current white-fill reader (works on the
    dock), keyed digit -> list of grayscale glyph bboxes (pad=1)."""
    old = {d: [_canon(t) for t in v]
           for d, v in load_digit_templates(
               Path("minigames/fishing/assets/digit_templates")).items()}
    inst = defaultdict(list)
    for cp in CORPUS:
        img = cv2.imread(cp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for patch, (x, y, w, h) in digit_components(img, binarize_white_fill):
            d, _ = match_digit(patch, old)
            if d is None:
                break
            y0, y1 = max(0, y - 1), min(gray.shape[0], y + h + 1)
            x0, x1 = max(0, x - 1), min(gray.shape[1], x + w + 1)
            inst[d].append(gray[y0:y1, x0:x1])
    return inst


def build_templates(inst):
    """digit -> (median_gray, mask) at the modal native padded size."""
    tmpl = {}
    for d in range(10):
        sz = Counter(g.shape for g in inst[d]).most_common(1)[0][0]
        stack = np.array([g for g in inst[d] if g.shape == sz], dtype=np.float32)
        med = np.median(stack, axis=0).astype(np.uint8)
        fill = (med >= 190).astype(np.uint8)
        mask = cv2.dilate(fill, np.ones((3, 3), np.uint8), 1) * 255
        tmpl[d] = (med, mask)
    return tmpl


def mzncc(patch, t, m):
    sel = m > 0
    a = t[sel].ravel().astype(np.float32)
    b = patch[sel].ravel().astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def read_zncc(crop, tmpl, thresh, dedup=4, max_gap=5, return_dets=False):
    gray = _to_gray(crop)
    H, W = gray.shape
    dets = []
    for d in range(10):
        t, m = tmpl[d]
        th, tw = t.shape
        if th > H or tw > W:
            continue
        for y in range(0, H - th + 1):
            for x in range(0, W - tw + 1):
                z = mzncc(gray[y:y + th, x:x + tw], t, m)
                if z >= thresh:
                    dets.append((x + tw / 2, x, x + tw, d, z))
    dets.sort(key=lambda r: -r[4])
    kept = []
    for r in dets:
        if all(abs(r[0] - k[0]) > dedup for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: r[0])
    if return_dets:
        return kept
    if not kept:
        return None
    digits = [kept[0]]
    for k in kept[1:]:
        if k[1] - digits[-1][2] <= max_gap:
            digits.append(k)
        else:
            break
    return int("".join(str(k[3]) for k in digits))


def main():
    inst = collect_instances()
    tmpl = build_templates(inst)
    print("instances:", {d: len(inst[d]) for d in range(10)})

    # True-digit peak ZNCC distribution: for each labeled instance, the peak.
    peaks = []
    for d in range(10):
        t, m = tmpl[d]
        for g in inst[d][:200]:
            best = -2
            th, tw = t.shape
            for y in range(0, max(1, g.shape[0] - th + 1)):
                for x in range(0, max(1, g.shape[1] - tw + 1)):
                    if y + th <= g.shape[0] and x + tw <= g.shape[1]:
                        best = max(best, mzncc(g[y:y + th, x:x + tw], t, m))
            peaks.append(best)
    peaks = np.array(peaks)
    print(f"true-digit peak ZNCC: min {peaks.min():.3f} p1 {np.percentile(peaks,1):.3f} "
          f"p5 {np.percentile(peaks,5):.3f} median {np.median(peaks):.3f}")

    # Threshold sweep over the full corpus.
    cache = []
    for cp in CORPUS:
        img = cv2.imread(cp)
        if img is None:
            continue
        if img.shape[0] > 60:
            cc = score_crop(img, find_cast_bar(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)))
            if cc is not None:
                cc = cv2.cvtColor(cc, cv2.COLOR_BGRA2BGR)
        else:
            cc = img
        if cc is None or cc.size == 0:
            continue
        cache.append((cp, cc, _read_pts(cc)))

    print(f"\nfull corpus: {len(cache)} crops")
    for thresh in (0.55, 0.6, 0.65, 0.7, 0.75):
        v2none = vdiff = none2v = same = bothnone = 0
        examples = []
        for cp, cc, old in cache:
            new = read_zncc(cc, tmpl, thresh)
            if old is None and new is None:
                bothnone += 1
            elif old is None:
                none2v += 1
            elif new is None:
                v2none += 1
                if len(examples) < 6:
                    examples.append(("V->None", Path(cp).name, old, new))
            elif old == new:
                same += 1
            else:
                vdiff += 1
                if len(examples) < 6:
                    examples.append(("V->DIFF", Path(cp).name, old, new))
        print(f"  thresh {thresh}: same {same}  V->None {v2none}  V->DIFF {vdiff}  "
              f"None->V {none2v}  bothNone {bothnone}  REGRESSIONS={v2none+vdiff}")
        for e in examples:
            print("      ", e)

    # Beach
    beach = cv2.imread(BEACH)
    bgra = cv2.cvtColor(beach, cv2.COLOR_BGR2BGRA)
    bar = find_cast_bar(bgra)
    bc = cv2.cvtColor(score_crop(bgra, bar), cv2.COLOR_BGRA2BGR)
    print(f"\nBEACH (expect 0): old={_read_pts(bc)} "
          f"new@0.7={read_zncc(bc, tmpl, 0.7)} dets={[(round(k[0],1),k[3],round(k[4],2)) for k in read_zncc(bc,tmpl,0.7,return_dets=True)]}")


if __name__ == "__main__":
    main()
