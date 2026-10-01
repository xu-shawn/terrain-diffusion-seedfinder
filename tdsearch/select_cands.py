"""Pick level-2 candidates from level-1 scan records.

Usage: python3 -m tdsearch.select_cands OUT.npy SCAN.npz [SCAN2.npz ...] [--top N] [--mode MODE]
"""

import argparse

import numpy as np

CAND_DTYPE = np.dtype([("seed", "<i8"), ("ci", "<i4"), ("cj", "<i4"), ("cscore", "<f4"), ("kind", "U8")])


def coarse_score(r, mode):
    """Rank water pixels by how much high ground crowds them."""
    height = np.maximum.reduce([r["m1"], 0.85 * r["m2"], 0.7 * r["m3"]])
    if mode == "iso":      # small inland water body: few water pixels around
        return np.where(r["nw5"] <= 4, height, 0.0)
    if mode == "iso9":
        return np.where(r["nw9"] <= 12, height, 0.0)
    if mode == "fjord":    # water inside a pixel whose own mean elevation is high
        return np.where(r["e"] > 500, r["e"] + 0.5 * height, 0.0)
    if mode == "coast":
        return height
    raise ValueError(mode)


def nms(recs, scores, top, radius=3):
    """Greedy: best first, drop anything within `radius` coarse pixels of a kept one (same seed)."""
    order = np.argsort(-scores)
    kept, seen = [], {}
    for idx in order:
        if scores[idx] <= 0 or len(kept) >= top:
            break
        s, ci, cj = int(recs["seed"][idx]), int(recs["ci"][idx]), int(recs["cj"][idx])
        if any(abs(ci - a) <= radius and abs(cj - b) <= radius for a, b in seen.get(s, ())):
            continue
        seen.setdefault(s, []).append((ci, cj))
        kept.append(idx)
    return np.asarray(kept, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("scans", nargs="+")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--modes", default="iso,iso9,fjord,coast")
    args = ap.parse_args()
    recs = np.concatenate([np.load(s)["recs"] for s in args.scans])
    out = []
    taken = set()
    for mode in args.modes.split(","):
        sc = coarse_score(recs, mode)
        for idx in nms(recs, sc, args.top):
            key = (int(recs["seed"][idx]), int(recs["ci"][idx]), int(recs["cj"][idx]))
            if key in taken:
                continue
            taken.add(key)
            out.append((key[0], key[1], key[2], sc[idx], mode))
    cands = np.array(out, dtype=CAND_DTYPE)
    np.save(args.out, cands)
    for mode in args.modes.split(","):
        sel = cands[cands["kind"] == mode]
        print(mode, len(sel), "score range", sel["cscore"].min() if len(sel) else None, sel["cscore"].max() if len(sel) else None)


if __name__ == "__main__":
    main()
