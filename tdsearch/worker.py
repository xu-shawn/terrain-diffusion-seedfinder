"""Production loop for one GPU: coarse-scan a chunk of seeds, then refine and score its best candidates.

Usage: python3 -m tdsearch.worker OUT_DIR K_START K_END [--chunk 60] [--thresh 3600]
Seeds are 1000*k + 7. Resumable: finished chunks leave a marker file.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import pipeline as P
from .analyze import analyze
from .scan_coarse import REC_DTYPE, score_tile, seed_of
from .select_cands import nms
from .synth import SyntheticMap


def coarse_candidate_score(r):
    """High ground right next to a water pixel; small bonus for inland (isolated) water."""
    return r["m1"] + 0.3 * np.maximum(r["e"], 0) + np.where(r["nw5"] <= 4, 800.0, 0.0)


def scan_chunk(models, ks, grid, batch=144):
    recs_all, pending, synths = [], [], {}
    tiles = [(a, b) for a in range(-grid, grid) for b in range(-grid, grid)]

    def flush():
        if not pending:
            return
        out = P.coarse_denoise(models, np.stack([p[3] for p in pending]), np.stack([p[4] for p in pending]))
        for k, (seed, ti, tj, _, _) in enumerate(pending):
            recs = score_tile(seed, ti, tj, out[k], 0.0, 2000.0)
            recs = recs[(recs["m1"] >= 2500) | (recs["nw5"] <= 4)]
            if len(recs):
                recs_all.append(recs)
        pending.clear()

    for k in ks:
        seed = seed_of(k)
        synths[seed] = SyntheticMap(seed)
        for ti, tj in tiles:
            cond, sample = P.coarse_tile_inputs(seed, synths[seed], ti, tj)
            pending.append((seed, ti, tj, cond, sample))
            if len(pending) >= batch:
                flush()
    flush()
    recs = np.concatenate(recs_all) if recs_all else np.zeros(0, dtype=REC_DTYPE)
    return recs, synths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("k_start", type=int)
    ap.add_argument("k_end", type=int)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=3600.0)
    ap.add_argument("--max-cands", type=int, default=40, help="per chunk")
    ap.add_argument("--half", type=int, default=640)
    ap.add_argument("--keep", type=float, default=2500.0)
    ap.add_argument("--batch", type=int, default=32, help="base-model batch; decoder uses a quarter")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "crops"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "recs"), exist_ok=True)
    models = P.Models()
    pool = ThreadPoolExecutor(max_workers=4)
    res_path = os.path.join(args.out_dir, "results.jsonl")
    t_start = time.time()

    def finish(elev, meta):
        summary, lakes = analyze(elev)
        best = lakes[0]["score"] if lakes else 0.0
        rec = dict(**meta, best=best, **summary, lakes=lakes[:4])
        if best >= args.keep:
            np.save(os.path.join(args.out_dir, "crops", f"s{meta['seed']}_{meta['ci']}_{meta['cj']}.npy"),
                    np.clip(np.rint(elev), -32768, 32767).astype(np.int16))
        return rec

    for c0 in range(args.k_start, args.k_end, args.chunk):
        marker = os.path.join(args.out_dir, "recs", f"chunk_{c0}.npz")
        if os.path.exists(marker):
            continue
        ks = range(c0, min(c0 + args.chunk, args.k_end))
        t0 = time.time()
        recs, synths = scan_chunk(models, ks, args.grid)
        t1 = time.time()
        sc = coarse_candidate_score(recs) if len(recs) else np.zeros(0)
        kept = nms(recs, np.where(sc >= args.thresh, sc, 0.0), args.max_cands, radius=2) if len(recs) else []
        futures = []
        for idx in kept:
            seed, ci, cj = int(recs["seed"][idx]), int(recs["ci"][idx]), int(recs["cj"][idx])
            i0, j0 = ci * 256 + 128 - args.half, cj * 256 + 128 - args.half
            elev = P.elev_region(models, seed, i0, j0, i0 + 2 * args.half, j0 + 2 * args.half,
                                 synth=synths[seed], batch=args.batch)
            meta = dict(seed=seed, ci=ci, cj=cj, i0=i0, j0=j0, half=args.half, cscore=float(sc[idx]))
            futures.append(pool.submit(finish, elev, meta))
        results = [f.result() for f in futures]
        with open(res_path, "a") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
        np.savez_compressed(marker, recs=recs)
        best = max([r["best"] for r in results] or [0.0])
        print(f"chunk k={c0}..{ks[-1]} scan {t1 - t0:.0f}s refine {time.time() - t1:.0f}s cands {len(kept)} "
              f"best {best:.0f} | elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
