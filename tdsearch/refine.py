"""Level-2: generate native-resolution terrain around coarse candidates and score it.

Usage: python3 -m tdsearch.refine CANDS.npy OUT_DIR [--half 640] [--start 0] [--count N]
CANDS.npy: structured array with fields seed, ci, cj (coarse pixel of the water body).
Appends one JSON line per candidate to OUT_DIR/results.jsonl and keeps the elevation crop
(int16 metres) for anything scoring above --keep.
"""

import argparse
import json
import os
import time

import numpy as np

from . import pipeline as P
from .analyze import analyze
from .synth import SyntheticMap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands")
    ap.add_argument("out_dir")
    ap.add_argument("--half", type=int, default=640)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--keep", type=float, default=3000.0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cands = np.load(args.cands)
    end = len(cands) if args.count is None else min(len(cands), args.start + args.count)
    models = P.Models()
    done = set()
    res_path = os.path.join(args.out_dir, "results.jsonl")
    if os.path.exists(res_path):
        with open(res_path) as f:
            done = {json.loads(line)["idx"] for line in f if line.strip()}
    synth_cache = {}
    t0 = time.time()
    n = 0
    with open(res_path, "a") as fout:
        for idx in range(args.start, end):
            if idx in done:
                continue
            c = cands[idx]
            seed, ci, cj = int(c["seed"]), int(c["ci"]), int(c["cj"])
            if seed not in synth_cache:
                if len(synth_cache) > 8:
                    synth_cache.clear()
                synth_cache[seed] = SyntheticMap(seed)
            i0 = ci * 256 + 128 - args.half
            j0 = cj * 256 + 128 - args.half
            elev = P.elev_region(models, seed, i0, j0, i0 + 2 * args.half, j0 + 2 * args.half,
                                 synth=synth_cache[seed], batch=args.batch)
            summary, lakes = analyze(elev)
            best = lakes[0]["score"] if lakes else 0.0
            rec = dict(idx=idx, seed=seed, ci=ci, cj=cj, i0=i0, j0=j0, half=args.half,
                       best=best, **summary, lakes=lakes[:4])
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if best >= args.keep:
                np.save(os.path.join(args.out_dir, f"elev_{idx:06d}.npy"),
                        np.clip(np.rint(elev), -32768, 32767).astype(np.int16))
            n += 1
            if n % 20 == 0:
                print(f"{n} done ({(time.time() - t0) / n:.2f}s each) last idx {idx} best {best:.0f}", flush=True)
    print("finished", n, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
