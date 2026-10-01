"""Properly powered test: can the noiseless Perlin conditioning map (stage 0, ~free) rule out tiles
before the coarse diffusion model runs?  Uses every tile scanned in production."""
import glob, json, sys
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
from tdsearch.synth import SyntheticMap
from tdsearch import pipeline as P

GRID = [(a, b) for a in range(-3, 3) for b in range(-3, 3)]

def tile_features(seed):
    sm = SyntheticMap(seed)
    out = []
    for ti, tj in GRID:
        i1, j1 = ti * 48, tj * 48
        xx, yy = np.meshgrid(np.arange(j1 + 10, j1 + 54, dtype=np.float32), np.arange(i1 + 10, i1 + 54, dtype=np.float32))
        e = sm.raw(0, xx, yy)                      # synthetic elevation (m) over interior +-6 px
        hi1 = ndi.maximum_filter(e, size=3); hi2 = ndi.maximum_filter(e, size=5); hi3 = ndi.maximum_filter(e, size=7)
        lo1 = ndi.minimum_filter(e, size=3); lo2 = ndi.minimum_filter(e, size=5)
        c = slice(4, -4)
        feats = [e[c, c].max(),                                        # f0: highest synthetic ground
                 (hi1 - lo1)[c, c].max(),                              # f1: relief within 3x3
                 (hi2 - lo2)[c, c].max(),                              # f2: relief within 5x5
                 np.where(lo2 <= 300, hi2, -1e4)[c, c].max(),          # f3: high ground within 2px of lowland
                 np.where(lo2 <= 0, hi3, -1e4)[c, c].max(),            # f4: high ground within 3px of synthetic sea
                 np.where(lo1 <= 600, hi1, -1e4)[c, c].max()]          # f5: high ground adjacent to low ground
        out.append((seed, ti, tj, *feats))
    return out

if __name__ == "__main__":
    recs = np.concatenate([np.load(f)["recs"] for f in glob.glob("prod/*/recs/chunk_*.npz")])
    seeds = sorted(set(int(s) for f in glob.glob("prod/*/recs/chunk_*.npz") for s in
                       [1000 * k + 7 for k in range(int(f.split("chunk_")[1].split(".")[0]), int(f.split("chunk_")[1].split(".")[0]) + 60)]))
    with Pool(11) as pool:
        feats = np.array([row for rows in pool.map(tile_features, seeds, chunksize=8) for row in rows])
    np.save("work/stage0_feats.npy", feats)
    key = {(int(s), int(a), int(b)): k for k, (s, a, b) in enumerate(feats[:, :3])}
    cs = recs["m1"] + 0.3 * np.maximum(recs["e"], 0)
    tile_cs = np.zeros(len(feats))
    for r, c in zip(recs, cs):
        k = key.get((int(r["seed"]), int(r["ci"]) // 48, int(r["cj"]) // 48))
        if k is not None: tile_cs[k] = max(tile_cs[k], c)
    # sites that were refined and turned out to have a strong lake-under-peak score
    good = np.zeros(len(feats), bool)
    for path in glob.glob("prod/*/results.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if r["best"] >= 3300:
                k = key.get((r["seed"], r["ci"] // 48, r["cj"] // 48))
                if k is not None: good[k] = True
    print("tiles", len(feats), "seeds", len(seeds))
    names = ["max synthetic elev", "relief 3x3", "relief 5x5", "high within 2px of lowland", "high within 3px of syn. sea", "high adjacent to low"]
    for label, pos in (("tile has coarse candidate >= 3600", tile_cs >= 3600), ("tile has coarse candidate >= 4000", tile_cs >= 4000),
                       ("tile has coarse candidate >= 4400", tile_cs >= 4400), ("tile holds a refined site scoring >= 3300", good)):
        print(f"\n{label}: {pos.sum()} positive tiles ({100 * pos.mean():.2f}%)")
        for j, n in enumerate(names):
            f = feats[:, 3 + j]
            line = f"  {n:28s}"
            for recall in (0.99, 0.95, 0.90, 0.80):
                thr = np.quantile(f[pos], 1 - recall)
                line += f" | recall {recall:.2f}: keep {100 * (f >= thr).mean():5.1f}%"
            print(line)
