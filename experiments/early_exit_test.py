"""How well does the sampler's denoised estimate after k steps predict the final coarse candidate score?"""
import math, numpy as np, time
from scipy import ndimage as ndi
from tdsearch import pipeline as P
from tdsearch.synth import SyntheticMap
f32 = np.float32

def denoise_with_previews(models, cond_mixed, sample):
    B = sample.shape[0]; sd2 = f32(0.25)
    conds = {f"cond_{k}": np.full(B, P.COND_VALS[k], dtype=np.float32) for k in range(5)}
    prev_x0 = None; previews = []
    for step in range(20):
        sigma, sigma_t = P.SIGMAS[step], P.SIGMAS[step + 1]
        sig2 = f32(sigma * sigma)
        cnoise = f32(math.atan(float(sigma) / 0.5)); c_in = f32(1.0) / f32(math.sqrt(float(sig2 + sd2)))
        out = models.run("coarse", {"x": np.concatenate([sample * c_in, cond_mixed], axis=1), "noise_labels": np.full(B, cnoise, dtype=np.float32), **conds})
        x0 = f32(sd2 / (sig2 + sd2)) * sample + f32(sigma * f32(0.5) / f32(math.sqrt(float(sig2 + sd2)))) * out
        previews.append(x0.copy())
        ratio = f32(sigma_t / sigma)
        if prev_x0 is None or step == 19:
            sample = ratio * sample - f32(ratio - f32(1)) * x0
        else:
            r0 = f32((-math.log(float(sigma)) + math.log(float(P.SIGMAS[step - 1]))) / (-math.log(float(sigma_t)) + math.log(float(sigma))))
            sample = ratio * sample + f32(-(ratio - f32(1))) * x0 + f32(f32(-0.5) * (ratio - f32(1))) * ((x0 - prev_x0) / r0)
        prev_x0 = x0
    return previews, sample

def denorm(x):
    out = (x / f32(0.5)) * P.MODEL_STDS[None, :, None, None] + P.MODEL_MEANS[None, :, None, None]
    out[:, 1] = out[:, 0] - out[:, 1]
    return out

def tile_score(o):
    """Best candidate score in the tile interior: high ground adjacent to a water pixel."""
    e = P.coarse_elev_m(o[0]); p5 = P.coarse_elev_m(o[1])
    m1 = ndi.maximum_filter(e, size=3, mode="nearest")
    cs = np.where((p5 < 0) & (e > -300), m1 + 0.3 * np.maximum(e, 0), 0)
    return float(cs[16:48, 16:48].max())

m = P.Models(want=("coarse",))
final_scores, prev_scores, syn_feats = [], [], []
t = time.time()
for seed in (900007, 901007, 902007, 903007, 904007, 905007):
    sm = SyntheticMap(seed)
    tiles = [(a, b) for a in range(-3, 3) for b in range(-3, 3)]
    ins = [P.coarse_tile_inputs(seed, sm, a, b) for a, b in tiles]
    cond = np.stack([c for c, _ in ins]); samp = np.stack([s for _, s in ins])
    previews, final = denoise_with_previews(m, cond, samp)
    fin = denorm(final)
    final_scores += [tile_score(o) for o in fin]
    prev_scores.append(np.array([[tile_score(o) for o in denorm(p)] for p in previews]))   # (20, 36)
    # synthetic-only prefilter feature: the same score computed on the noiseless synthetic map
    for a, b in tiles:
        syn = sm.sample(b * 48, a * 48, b * 48 + 64, a * 48 + 64)
        e = P.coarse_elev_m(syn[0]); m1 = ndi.maximum_filter(e, size=3, mode="nearest"); m2 = ndi.maximum_filter(e, size=5, mode="nearest")
        lo = ndi.minimum_filter(e, size=3, mode="nearest")
        syn_feats.append([float(np.where(lo < 150, m2, 0)[14:50, 14:50].max()), float(e[12:52, 12:52].max()), float(np.where(e < 100, m1, 0)[14:50, 14:50].max())])
    print(seed, time.time() - t, flush=True)
np.savez("work/early_exit.npz", final=np.array(final_scores), prev=np.concatenate(prev_scores, axis=1), syn=np.array(syn_feats))
