"""Level-1 scan: run the coarse model over many seeds and log high ground next to sea-level water.

Only the interior 32x32 of each 64x64 coarse tile is scored: there the tile alone determines the
mod's blended coarse map, so the logged values are exact.

Usage: python3 -m tdsearch.scan_coarse OUT.npz SEED_K_START SEED_K_COUNT [--grid 3] [--prefilter M]
Seeds are 1000*k + 7.
"""

import argparse
import time

import numpy as np
from scipy import ndimage as ndi

from . import pipeline as P
from .synth import SyntheticMap

INT0, INT1 = 16, 48  # interior of a tile that no neighbouring tile overlaps

REC_DTYPE = np.dtype([
    ("seed", "<i8"), ("ci", "<i4"), ("cj", "<i4"),
    ("e", "<f4"), ("p5", "<f4"),          # water pixel: mean elevation and 5th percentile (m)
    ("m1", "<f4"), ("m2", "<f4"), ("m3", "<f4"),  # max mean elevation within Chebyshev radius 1/2/3
    ("r1", "<f4"),                        # max (mean - p5) within radius 1: ruggedness
    ("nw3", "<i2"), ("nw5", "<i2"), ("nw9", "<i2"),  # water-pixel counts in 3x3 / 5x5 / 9x9
    ("syn_max", "<f4"),
])


def seed_of(k):
    return 1000 * k + 7


def score_tile(seed, ti, tj, out, syn_max, min_m2):
    e = P.coarse_elev_m(out[0])
    p5 = P.coarse_elev_m(out[1])
    water = p5 < 0
    m1 = ndi.maximum_filter(e, size=3, mode="nearest")
    m2 = ndi.maximum_filter(e, size=5, mode="nearest")
    m3 = ndi.maximum_filter(e, size=7, mode="nearest")
    r1 = ndi.maximum_filter(e - p5, size=3, mode="nearest")
    wf = water.astype(np.float32)
    nw3 = ndi.uniform_filter(wf, size=3, mode="nearest") * 9
    nw5 = ndi.uniform_filter(wf, size=5, mode="nearest") * 25
    nw9 = ndi.uniform_filter(wf, size=9, mode="nearest") * 81
    sel = np.zeros_like(water)
    sel[INT0:INT1, INT0:INT1] = True
    sel &= water & (e > -300) & (m2 >= min_m2)
    recs = np.zeros(int(sel.sum()), dtype=REC_DTYPE)
    ii, jj = np.nonzero(sel)
    recs["seed"] = seed
    recs["ci"] = ti * P.COARSE_ST + ii
    recs["cj"] = tj * P.COARSE_ST + jj
    for name, arr in (("e", e), ("p5", p5), ("m1", m1), ("m2", m2), ("m3", m3), ("r1", r1),
                      ("nw3", np.rint(nw3)), ("nw5", np.rint(nw5)), ("nw9", np.rint(nw9))):
        recs[name] = arr[ii, jj]
    recs["syn_max"] = syn_max
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("k_start", type=int)
    ap.add_argument("k_count", type=int)
    ap.add_argument("--grid", type=int, default=3, help="tiles ti,tj in [-grid, grid)")
    ap.add_argument("--min-m2", type=float, default=2000.0)
    ap.add_argument("--prefilter", type=float, default=None,
                    help="skip tiles whose synthetic elevation never reaches this many metres")
    ap.add_argument("--batch", type=int, default=144)
    args = ap.parse_args()

    models = P.Models(want=("coarse",))
    grid = [(a, b) for a in range(-args.grid, args.grid) for b in range(-args.grid, args.grid)]
    all_recs, tile_stats = [], []
    pending = []  # (seed, ti, tj, cond, sample, syn_max)
    t0 = time.time()
    n_tiles = n_skipped = 0

    def flush():
        nonlocal n_tiles
        if not pending:
            return
        out = P.coarse_denoise(models, np.stack([p[3] for p in pending]), np.stack([p[4] for p in pending]))
        for k, (seed, ti, tj, _, _, syn_max) in enumerate(pending):
            recs = score_tile(seed, ti, tj, out[k], syn_max, args.min_m2)
            e = P.coarse_elev_m(out[k][0])
            tile_stats.append((seed, ti, tj, syn_max, e[INT0:INT1, INT0:INT1].max(), len(recs),
                               recs["m2"].max() if len(recs) else 0.0))
            if len(recs):
                all_recs.append(recs)
        n_tiles += len(pending)
        pending.clear()

    def save():
        recs = np.concatenate(all_recs) if all_recs else np.zeros(0, dtype=REC_DTYPE)
        np.savez(args.out, recs=recs, tile_stats=np.asarray(tile_stats, dtype=np.float64),
                 k_start=args.k_start, k_done=kk + 1, grid=args.grid)

    for kk in range(args.k_count):
        seed = seed_of(args.k_start + kk)
        synth = SyntheticMap(seed)
        for ti, tj in grid:
            # Synthetic elevation over the tile: a cheap predictor of where the coarse model puts mountains.
            i1, j1 = ti * P.COARSE_ST, tj * P.COARSE_ST
            xx, yy = np.meshgrid(np.arange(j1, j1 + 64, 2, dtype=np.float32), np.arange(i1, i1 + 64, 2, dtype=np.float32))
            syn_e = synth.raw(0, xx, yy)
            syn_max = float(syn_e.max())
            if args.prefilter is not None and (syn_max < args.prefilter or syn_e.min() > 0):
                n_skipped += 1
                continue
            cond, sample = P.coarse_tile_inputs(seed, synth, ti, tj)
            pending.append((seed, ti, tj, cond, sample, syn_max))
            if len(pending) >= args.batch:
                flush()
        if (kk + 1) % 50 == 0:
            flush()
            save()
            n = sum(len(r) for r in all_recs)
            print(f"seeds {kk + 1}/{args.k_count} tiles {n_tiles} skipped {n_skipped} recs {n} "
                  f"({(time.time() - t0):.0f}s, {n_tiles / (time.time() - t0):.1f} tiles/s)", flush=True)
    flush()
    save()
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
