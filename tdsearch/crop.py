"""Generate a native-resolution (30 m) elevation crop around a coarse pixel, as refine/worker keep them.

Usage: python3 -m tdsearch.crop SEED CI CJ OUT.npy [--half 640]
Writes int16 metres; prints the crop's native origin (i0, j0), which finalize/shots need.
"""

import argparse

import numpy as np

from . import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("ci", type=int)
    ap.add_argument("cj", type=int)
    ap.add_argument("out")
    ap.add_argument("--half", type=int, default=640)
    args = ap.parse_args()
    i0, j0 = args.ci * 256 + 128 - args.half, args.cj * 256 + 128 - args.half
    elev = P.elev_region(P.Models(), args.seed, i0, j0, i0 + 2 * args.half, j0 + 2 * args.half, verbose=True)
    np.save(args.out, np.clip(np.rint(elev), -32768, 32767).astype(np.int16))
    print(f"origin i0={i0} j0={j0}  (blocks at scale 6: Z {i0 * 6}, X {j0 * 6});  peak {elev.max():.0f} m")


if __name__ == "__main__":
    main()
