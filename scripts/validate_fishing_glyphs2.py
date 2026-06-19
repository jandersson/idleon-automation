"""Fast validation of the masked-ZNCC fishing score reader (#82), v2.

Uses the vectorized exact-ZNCC map (verified == masked_match_confidence) so the
full corpus runs in seconds. Reports old-vs-new regressions, recoveries, and
per-session monotonicity (the old-reader-independent correctness signal: a
session's score never decreases, so a non-monotonic new read is a hallucination).
"""
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.score_digits import binarize_white_fill, make_pts_reader
from minigames.fishing.detector import find_cast_bar
from minigames.fishing.score import score_crop

# The pre-#82 reader, constructed explicitly (score._read_pts is now the masked-
# ZNCC reader) so the comparison stays "old white-fill binarize vs new ZNCC".
_read_pts = make_pts_reader(
    Path(__file__).parent.parent / "minigames/fishing/assets/digit_templates",
    binarize=binarize_white_fill)

GLYPHS = Path(__file__).parent.parent / "minigames/fishing/assets/digit_glyphs"
CORPUS = sorted(glob.glob(str(Path(__file__).parent.parent /
        "minigames/fishing/assets/captures/botrun_2026061*/score_c*.png")))
BEACH = r"C:\Program Files (x86)\Steam\userdata\153875\760\remote\1476970\screenshots\20260618155212_1.jpg"


def load_tmpl():
    t = {}
    for d in range(10):
        g = cv2.imread(str(GLYPHS / f"{d}.png"), 0)
        m = cv2.imread(str(GLYPHS / f"{d}_mask.png"), 0)
        t[d] = (g, m)
    return t


def zncc_map(I, tmpl, mask):
    I = I.astype(np.float32)
    T = tmpl.astype(np.float32)
    M = (mask > 0).astype(np.float32)
    N = float(M.sum())
    Tm = float((M * T).sum() / N)
    Tc = M * (T - Tm)
    normT = float(np.sqrt((Tc * Tc).sum()))
    if normT == 0 or I.shape[0] < T.shape[0] or I.shape[1] < T.shape[1]:
        return None
    cross = cv2.matchTemplate(I, Tc, cv2.TM_CCORR)
    sumI = cv2.matchTemplate(I, M, cv2.TM_CCORR)
    sumI2 = cv2.matchTemplate(I * I, M, cv2.TM_CCORR)
    varI = sumI2 - (sumI * sumI) / N
    den = normT * np.sqrt(np.maximum(varI, 0))
    out = np.zeros_like(cross)
    nz = den > 1e-6
    out[nz] = cross[nz] / den[nz]
    return out


def _to_gray(crop):
    if crop.ndim == 2:
        return crop
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def read_zncc(crop, tmpl, thresh=0.6, dedup=4, max_gap=5, max_peaks=6, return_dets=False):
    gray = _to_gray(crop)
    dets = []
    for d in range(10):
        t, m = tmpl[d]
        th, tw = t.shape
        zm = zncc_map(gray, t, m)
        if zm is None:
            continue
        work = zm.copy()
        for _ in range(max_peaks):
            py, px = np.unravel_index(int(np.argmax(work)), work.shape)
            z = float(work[py, px])
            if z < thresh:
                break
            dets.append((px + tw / 2, px, px + tw, d, z))
            work[max(0, py - th // 2):py + th // 2 + 1,
                 max(0, px - tw // 2):px + tw // 2 + 1] = -2
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


def crop_of(img):
    if img.shape[0] > 60:
        cc = score_crop(img, find_cast_bar(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)))
        return cv2.cvtColor(cc, cv2.COLOR_BGRA2BGR) if cc is not None else None
    return img


def main():
    tmpl = load_tmpl()
    cache = []
    for cp in CORPUS:
        img = cv2.imread(cp)
        if img is None:
            continue
        cc = crop_of(img)
        if cc is None or cc.size == 0:
            continue
        cache.append((cp, cc, _read_pts(cc)))
    print(f"corpus: {len(cache)} crops")

    for thresh in (0.6, 0.62, 0.65):
        v2none = vdiff = none2v = same = bothnone = 0
        ex = []
        for cp, cc, old in cache:
            new = read_zncc(cc, tmpl, thresh)
            if old is None and new is None:
                bothnone += 1
            elif old is None:
                none2v += 1
            elif new is None:
                v2none += 1
                ex.append(("V->None", Path(cp).name, old, new))
            elif old == new:
                same += 1
            else:
                vdiff += 1
                ex.append(("V->DIFF", Path(cp).name, old, new))
        print(f"\nthresh {thresh}: same {same} V->None {v2none} V->DIFF {vdiff} "
              f"None->V {none2v} bothNone {bothnone}  REGRESSIONS={v2none+vdiff}")
        for e in ex:
            print("   ", e)

    # Per-session monotonicity of the NEW reader (old-reader-independent).
    print("\n=== per-session monotonicity (new reader @0.6) ===")
    by_run = defaultdict(list)
    for cp, cc, old in cache:
        run = Path(cp).parent.name
        m = re.search(r"score_c(\d+)_(before|after)", Path(cp).name)
        if not m:
            continue
        by_run[run].append((int(m.group(1)), m.group(2) == "after", cc, Path(cp).name))
    total_drops = 0
    for run, items in sorted(by_run.items()):
        items.sort(key=lambda t: (t[0], t[1]))
        seq = [(name, read_zncc(cc, tmpl, 0.6)) for _, _, cc, name in items]
        vals = [(n, v) for n, v in seq if v is not None]
        drops = [(vals[i - 1], vals[i]) for i in range(1, len(vals)) if vals[i][1] < vals[i - 1][1]]
        if drops:
            total_drops += len(drops)
            print(f"  {run}: {len(drops)} drop(s) of {len(vals)} reads; first: {drops[0]}")
    print(f"total non-monotonic drops across sessions: {total_drops}")

    beach = cv2.imread(BEACH)
    bc = crop_of(beach)
    print(f"\nBEACH (expect 0): old={_read_pts(bc)} new={read_zncc(bc, tmpl, 0.6)}")


if __name__ == "__main__":
    main()
