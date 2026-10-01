"""Batched numpy/onnxruntime port of the mod's WorldPipeline (coarse -> latent -> decoder -> elevation).

Mirrors terrain-diffusion-mc v2.2.0 WorldPipeline.java. Coordinates: i = row = Minecraft Z axis,
j = column = Minecraft X axis. One coarse pixel = 256 native pixels; one native pixel = 30 m.
"""

import json
import math
import os

import numpy as np
import onnxruntime as ort

from .rng import noise_patch
from .synth import MODELS_DIR, SyntheticMap

f32 = np.float32

with open(os.path.join(MODELS_DIR, "world_pipeline_config.json")) as _f:
    CFG = json.load(_f)

LC = CFG["latent_compression"]
SIGMA_DATA = f32(0.5)
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0

COARSE_S, COARSE_ST = 64, 48
LATENT_S, LATENT_ST = 64, 32
DEC_S, DEC_ST = 256, 192

MODEL_MEANS = np.asarray(CFG["coarse_means"], dtype=np.float32)
MODEL_STDS = np.asarray(CFG["coarse_stds"], dtype=np.float32)
COND_MEANS = np.asarray([14.99, 11.65, 15.87, 619.26, 833.12, 69.40, 0.66], dtype=np.float32)
COND_STDS = np.asarray([21.72, 21.78, 10.40, 452.29, 738.09, 34.59, 0.47], dtype=np.float32)
LOWFREQ_MEAN, LOWFREQ_STD = f32(-31.4), f32(38.6)
RESIDUAL_MEAN, RESIDUAL_STD = f32(CFG["residual_mean"]), f32(CFG["residual_std"])
COND_SNR = np.asarray(CFG["cond_snr"], dtype=np.float32)
COND_VALS = np.log(COND_SNR.astype(np.float64) / 8.0).astype(np.float32)
HISTOGRAM_RAW = np.asarray(CFG["histogram_raw"] or [0, 0, 0, 0, 0], dtype=np.float32)

COND_DIMS = [16, 16, 4, 16, 5, 1]
_C = f32(math.sqrt(sum(COND_DIMS) * len(COND_DIMS)))
MP_SCALES = [f32(_C / f32(math.sqrt(n)) / len(COND_DIMS)) for n in COND_DIMS]


def weight_window(size):
    mid = f32((size - 1) / 2.0)
    eps = f32(1e-3)
    r = np.arange(size, dtype=np.float32)
    w1 = f32(1) - (f32(1) - eps) * np.minimum(f32(1), np.abs(r - mid) / mid)
    return (w1[:, None] * w1[None, :]).astype(np.float32)


WW_COARSE = weight_window(COARSE_S)
WW_LATENT = weight_window(LATENT_S)
WW_DEC = weight_window(DEC_S)


def karras_sigmas(n):
    min_inv = f32(SIGMA_MIN ** (1.0 / RHO))
    max_inv = f32(SIGMA_MAX ** (1.0 / RHO))
    sig = np.zeros(n + 1, dtype=np.float32)
    for i in range(n):
        t = f32(f32(i) / f32(n - 1))
        inv = f32(max_inv + t * f32(min_inv - max_inv))
        sig[i] = f32(float(inv) ** RHO)
    return sig


SIGMAS = karras_sigmas(20)
_COS_C = np.cos(np.arctan(COND_SNR.astype(np.float64))).astype(np.float32)
_SIN_C = np.sin(np.arctan(COND_SNR.astype(np.float64))).astype(np.float32)


def _ceil_div(a, b):
    return -((-a) // b)


def windows_for(lo, hi, size, stride):
    """Inclusive window-index range whose extents [w*stride, w*stride+size) intersect [lo, hi)."""
    return _ceil_div(lo - size + 1, stride), (hi - 1) // stride


class Models:
    def __init__(self, providers=None, want=("coarse", "base", "decoder"), threads=0):
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            if providers[0] == "CUDAExecutionProvider":
                # Grow the GPU arena only by what each request needs (matters on 16 GB cards).
                providers[0] = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = threads
        self.providers = providers
        self.sess = {}
        for name in want:
            self.sess[name] = ort.InferenceSession(
                os.path.join(MODELS_DIR, f"{name}_model.onnx"), sess_options=opts, providers=providers)

    def run(self, name, feed):
        return self.sess[name].run(None, feed)[0]


# ---------------------------------------------------------------------------------------------
# Coarse stage
# ---------------------------------------------------------------------------------------------

def coarse_tile_inputs(seed, synth, ti, tj):
    """Conditioning image (5,S,S) and initial sample (6,S,S) for coarse tile (ti, tj)."""
    S = COARSE_S
    i1, j1 = ti * COARSE_ST, tj * COARSE_ST
    syn = synth.sample(j1, i1, j1 + S, i1 + S)
    syn[1] = np.where(syn[1] <= f32(20), (syn[1] - f32(20)) * f32(1.25) + f32(20), syn[1])
    idx = [0, 2, 3, 4, 5]
    cond = (syn - MODEL_MEANS[idx, None, None]) / MODEL_STDS[idx, None, None]
    cond_noise = noise_patch(seed, i1, j1, S, S, 5, S, S)
    cond_mixed = _COS_C[:, None, None] * cond + _SIN_C[:, None, None] * cond_noise
    sample = noise_patch(seed + 1, i1, j1, S, S, 6, S, S) * SIGMAS[0]
    return cond_mixed.astype(np.float32), sample.astype(np.float32)


def coarse_denoise(models, cond_mixed, sample):
    """20-step DPM-Solver++ (2M, midpoint, Karras). Batched: cond (B,5,S,S), sample (B,6,S,S).

    Returns denormalised (B,6,S,S): [elev_sqrt, p5, temp, temp_std, precip, precip_cv]."""
    B = sample.shape[0]
    sd2 = f32(SIGMA_DATA * SIGMA_DATA)
    conds = {f"cond_{k}": np.full(B, COND_VALS[k], dtype=np.float32) for k in range(5)}
    prev_x0 = None
    for step in range(20):
        sigma, sigma_t = SIGMAS[step], SIGMAS[step + 1]
        sig2 = f32(sigma * sigma)
        cnoise = f32(math.atan(float(sigma) / float(SIGMA_DATA)))
        c_in = f32(1.0) / f32(math.sqrt(float(sig2 + sd2)))
        x_in = np.concatenate([sample * c_in, cond_mixed], axis=1)
        out = models.run("coarse", {"x": x_in, "noise_labels": np.full(B, cnoise, dtype=np.float32), **conds})
        c_skip = f32(sd2 / (sig2 + sd2))
        c_out = f32(sigma * SIGMA_DATA / f32(math.sqrt(float(sig2 + sd2))))
        x0 = c_skip * sample + c_out * out
        ratio = f32(sigma_t / sigma)
        if prev_x0 is None or step == 19:
            sample = ratio * sample - f32(ratio - f32(1)) * x0
        else:
            l_t, l_s0, l_s1 = -math.log(float(sigma_t)), -math.log(float(sigma)), -math.log(float(SIGMAS[step - 1]))
            r0 = f32((l_s0 - l_s1) / (l_t - l_s0))
            d1 = (x0 - prev_x0) / r0
            sample = ratio * sample + f32(-(ratio - f32(1))) * x0 + f32(f32(-0.5) * (ratio - f32(1))) * d1
        prev_x0 = x0
    out = (sample / SIGMA_DATA) * MODEL_STDS[None, :, None, None] + MODEL_MEANS[None, :, None, None]
    out[:, 1] = out[:, 0] - out[:, 1]
    return out.astype(np.float32)


def coarse_tiles(models, seed, synth, tiles, batch=64):
    """Compute denormalised coarse tiles. tiles: list of (ti, tj). Returns dict -> (6,S,S)."""
    result = {}
    for b0 in range(0, len(tiles), batch):
        chunk = tiles[b0:b0 + batch]
        ins = [coarse_tile_inputs(seed, synth, ti, tj) for ti, tj in chunk]
        out = coarse_denoise(models, np.stack([c for c, _ in ins]), np.stack([s for _, s in ins]))
        for k, t in enumerate(chunk):
            result[t] = out[k]
    return result


def blend_coarse(tile_dict, ci0, cj0, ci1, cj1):
    """Weighted blend of coarse tiles over [ci0,ci1) x [cj0,cj1). Returns (acc(6,H,W), weight(H,W))."""
    H, W = ci1 - ci0, cj1 - cj0
    acc = np.zeros((6, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    S, ST = COARSE_S, COARSE_ST
    ti0, ti1 = windows_for(ci0, ci1, S, ST)
    tj0, tj1 = windows_for(cj0, cj1, S, ST)
    for ti in range(ti0, ti1 + 1):
        for tj in range(tj0, tj1 + 1):
            tile = tile_dict[(ti, tj)]
            a0, a1 = max(ci0, ti * ST), min(ci1, ti * ST + S)
            b0, b1 = max(cj0, tj * ST), min(cj1, tj * ST + S)
            ws = WW_COARSE[a0 - ti * ST:a1 - ti * ST, b0 - tj * ST:b1 - tj * ST]
            acc[:, a0 - ci0:a1 - ci0, b0 - cj0:b1 - cj0] += tile[:, a0 - ti * ST:a1 - ti * ST, b0 - tj * ST:b1 - tj * ST] * ws
            wsum[a0 - ci0:a1 - ci0, b0 - cj0:b1 - cj0] += ws
    return acc, wsum


def coarse_region(models, seed, ci0, cj0, ci1, cj1, synth=None, batch=64, tile_cache=None):
    """Exact blended coarse map (6, H, W) over coarse-pixel range, as the mod's getCoarseSlice / weight."""
    synth = synth or SyntheticMap(seed)
    ti0, ti1 = windows_for(ci0, ci1, COARSE_S, COARSE_ST)
    tj0, tj1 = windows_for(cj0, cj1, COARSE_S, COARSE_ST)
    tiles = [(ti, tj) for ti in range(ti0, ti1 + 1) for tj in range(tj0, tj1 + 1)]
    tile_cache = {} if tile_cache is None else tile_cache
    missing = [t for t in tiles if t not in tile_cache]
    tile_cache.update(coarse_tiles(models, seed, synth, missing, batch=batch))
    acc, wsum = blend_coarse(tile_cache, ci0, cj0, ci1, cj1)
    return acc / wsum[None]


def coarse_elev_m(coarse):
    """Signed-sqrt coarse channel -> metres."""
    return np.sign(coarse) * coarse * coarse


# ---------------------------------------------------------------------------------------------
# Latent stage
# ---------------------------------------------------------------------------------------------

def latent_conditioning(coarse_slice):
    """58-dim conditioning from an unweighted (6,4,4) coarse slice."""
    norm = (coarse_slice - COND_MEANS[:6, None, None]) / COND_STDS[:6, None, None]
    norm = np.nan_to_num(norm, nan=0.0).astype(np.float32)
    mask_norm = f32((f32(1.0) - COND_MEANS[6]) / COND_STDS[6])
    climate_means = np.array([norm[2 + ch, 1:3, 1:3].sum(dtype=np.float32) / f32(4) for ch in range(4)], dtype=np.float32)
    noise_level_norm = f32((0.0 - 0.5) * math.sqrt(12.0))
    return np.concatenate([
        norm[0].ravel() * MP_SCALES[0],
        norm[1].ravel() * MP_SCALES[1],
        climate_means * MP_SCALES[2],
        np.full(16, mask_norm, dtype=np.float32) * MP_SCALES[3],
        HISTOGRAM_RAW * MP_SCALES[4],
        np.array([noise_level_norm * MP_SCALES[5]], dtype=np.float32),
    ]).astype(np.float32)


def _latent_pass(models, seed, coarse, c_org, wins, prev, prev_org, t, seed_offset, batch):
    """Run the base model for latent windows. Returns dict (wi,wj) -> (5,S,S) unweighted sample.

    coarse: (6,Hc,Wc) with origin c_org; prev: accumulated (5,H,W)/weight array of the previous
    pass (already divided by weight) with origin prev_org in latent pixels, or None."""
    S, ST = LATENT_S, LATENT_ST
    cos_t, sin_t = f32(math.cos(t)), f32(math.sin(t))
    out = {}
    for b0 in range(0, len(wins), batch):
        chunk = wins[b0:b0 + batch]
        xts, conds = [], []
        for wi, wj in chunk:
            ci, cj = wi - 1 - c_org[0], wj - 1 - c_org[1]
            conds.append(latent_conditioning(coarse[:, ci:ci + 4, cj:cj + 4]))
            i1, j1 = wi * ST, wj * ST
            noise = noise_patch(seed + seed_offset, i1, j1, S, S, 5, S, S)
            xt = sin_t * (noise * SIGMA_DATA)
            if prev is not None:
                pi, pj = i1 - prev_org[0], j1 - prev_org[1]
                xt = cos_t * (prev[:, pi:pi + S, pj:pj + S] * SIGMA_DATA) + xt
            xts.append(xt.astype(np.float32))
        xts = np.stack(xts)
        pred = -models.run("base", {
            "x": (xts / SIGMA_DATA).astype(np.float32),
            "noise_labels": np.full(len(chunk), f32(t), dtype=np.float32),
            "cond_0": np.stack(conds)})
        new = (cos_t * xts - sin_t * SIGMA_DATA * pred) / SIGMA_DATA
        for k, w in enumerate(chunk):
            out[w] = new[k].astype(np.float32)
    return out


def _accumulate(win_dict, lo, hi, size, stride, ww, channels):
    """Blend windows over pixel box lo=(i,j) .. hi=(i,j). Returns value/weight (channels,H,W)."""
    H, W = hi[0] - lo[0], hi[1] - lo[1]
    acc = np.zeros((channels, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    for (wi, wj), tile in win_dict.items():
        a0, a1 = max(lo[0], wi * stride), min(hi[0], wi * stride + size)
        b0, b1 = max(lo[1], wj * stride), min(hi[1], wj * stride + size)
        if a0 >= a1 or b0 >= b1:
            continue
        ws = ww[a0 - wi * stride:a1 - wi * stride, b0 - wj * stride:b1 - wj * stride]
        acc[:, a0 - lo[0]:a1 - lo[0], b0 - lo[1]:b1 - lo[1]] += tile[:, a0 - wi * stride:a1 - wi * stride, b0 - wj * stride:b1 - wj * stride] * ws
        wsum[a0 - lo[0]:a1 - lo[0], b0 - lo[1]:b1 - lo[1]] += ws
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum[None] > f32(1e-6), acc / wsum[None], f32(0)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------------------------

def bilinear_resize(src, dst_h, dst_w):
    """LaplacianUtils.bilinearResize (align_corners=False, edge clamp)."""
    src_h, src_w = src.shape

    def axis(dst_n, src_n):
        pos = (np.arange(dst_n, dtype=np.float32) + f32(0.5)) * f32(src_n) / f32(dst_n) - f32(0.5)
        i0 = np.floor(pos).astype(np.int64)
        w = (pos - i0.astype(np.float32)).astype(np.float32)
        return np.clip(i0, 0, src_n - 1), np.clip(i0 + 1, 0, src_n - 1), w

    r0, r1, wr = axis(dst_h, src_h)
    c0, c1, wc = axis(dst_w, src_w)
    top = src[r0][:, c0] * (f32(1) - wc)[None, :] + src[r0][:, c1] * wc[None, :]
    bot = src[r1][:, c0] * (f32(1) - wc)[None, :] + src[r1][:, c1] * wc[None, :]
    return (top * (f32(1) - wr)[:, None] + bot * wr[:, None]).astype(np.float32)


def bilinear_resize_extrapolated(src, dst_h, dst_w):
    s_h, s_w = src.shape
    padded = np.zeros((s_h + 2, s_w + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = src
    padded[0, 1:-1] = 2 * src[0] - src[1]
    padded[-1, 1:-1] = 2 * src[-1] - src[-2]
    padded[:, 0] = 2 * padded[:, 1] - padded[:, 2]
    padded[:, -1] = 2 * padded[:, -2] - padded[:, -3]
    new_h = int(round(dst_h + 2.0 * dst_h / s_h))
    new_w = int(round(dst_w + 2.0 * dst_w / s_w))
    resized = bilinear_resize(padded, new_h, new_w)
    pad_h, pad_w = int(round(dst_h / s_h)), int(round(dst_w / s_w))
    return resized[pad_h:pad_h + dst_h, pad_w:pad_w + dst_w]


def gaussian_blur_clamped(src, sigma):
    ks = (int(sigma * 2) // 2) * 2 + 1
    half = ks // 2
    x = np.arange(ks, dtype=np.float32) - half
    k = np.exp(-0.5 * x.astype(np.float64) ** 2 / (sigma * sigma)).astype(np.float32)
    k = k / k.sum(dtype=np.float32)
    pad = np.pad(src, ((0, 0), (half, half)), mode="edge")
    tmp = sum(pad[:, i:i + src.shape[1]] * k[i] for i in range(ks))
    pad = np.pad(tmp, ((half, half), (0, 0)), mode="edge")
    return sum(pad[i:i + src.shape[0], :] * k[i] for i in range(ks)).astype(np.float32)


def elev_region(models, seed, i1, j1, i2, j2, synth=None, batch=16, coarse_cache=None, verbose=False):
    """Native-resolution elevation in metres over [i1,i2) x [j1,j2), as WorldPipeline.get(...)[0]."""
    synth = synth or SyntheticMap(seed)
    sigma = 5.0
    ks = (int(sigma * 2) // 2) * 2 + 1
    pad_hr = (ks // 2 + 1) * LC
    pi1 = ((i1 - pad_hr) // LC) * LC
    pj1 = ((j1 - pad_hr) // LC) * LC
    pi2 = _ceil_div(i2 + pad_hr, LC) * LC
    pj2 = _ceil_div(j2 + pad_hr, LC) * LC

    # Decoder windows covering the padded box, and the latent box they read.
    d_i = windows_for(pi1, pi2, DEC_S, DEC_ST)
    d_j = windows_for(pj1, pj2, DEC_S, DEC_ST)
    dls, dlst = DEC_S // LC, DEC_ST // LC
    l_lo = (min(d_i[0] * dlst, pi1 // LC), min(d_j[0] * dlst, pj1 // LC))
    l_hi = (max(d_i[1] * dlst + dls, pi2 // LC), max(d_j[1] * dlst + dls, pj2 // LC))

    # Step-latent windows covering that latent box; init windows are one ring wider.
    s_i = windows_for(l_lo[0], l_hi[0], LATENT_S, LATENT_ST)
    s_j = windows_for(l_lo[1], l_hi[1], LATENT_S, LATENT_ST)
    step_wins = [(a, b) for a in range(s_i[0], s_i[1] + 1) for b in range(s_j[0], s_j[1] + 1)]
    init_wins = [(a, b) for a in range(s_i[0] - 1, s_i[1] + 2) for b in range(s_j[0] - 1, s_j[1] + 2)]

    c_org = (s_i[0] - 2, s_j[0] - 2)
    c_end = (s_i[1] + 1 + 3, s_j[1] + 1 + 3)
    coarse = coarse_region(models, seed, c_org[0], c_org[1], c_end[0], c_end[1], synth=synth, tile_cache=coarse_cache)

    if verbose:
        print(f"  latent windows: {len(init_wins)} init + {len(step_wins)} step; decoder windows: "
              f"{(d_i[1] - d_i[0] + 1) * (d_j[1] - d_j[0] + 1)}", flush=True)

    t_init = math.atan(float(f32(SIGMA_MAX)) / float(SIGMA_DATA))
    init = _latent_pass(models, seed, coarse, c_org, init_wins, None, None, f32(t_init), 5819, batch)
    i_lo = ((s_i[0]) * LATENT_ST, (s_j[0]) * LATENT_ST)
    i_hi = ((s_i[1]) * LATENT_ST + LATENT_S, (s_j[1]) * LATENT_ST + LATENT_S)
    init_map = _accumulate(init, i_lo, i_hi, LATENT_S, LATENT_ST, WW_LATENT, 5)
    del init

    t_step = math.atan(float(f32(0.35)) / float(SIGMA_DATA))
    step = _latent_pass(models, seed, coarse, c_org, step_wins, init_map, i_lo, f32(t_step), 5820, batch)
    latents = _accumulate(step, l_lo, l_hi, LATENT_S, LATENT_ST, WW_LATENT, 5)
    del step, init_map

    # Decoder: one flow-matching step per 256x256 window.
    t_dec = f32(math.atan(float(f32(SIGMA_MAX)) / float(SIGMA_DATA)))
    cos_t, sin_t = f32(math.cos(t_dec)), f32(math.sin(t_dec))
    dec_wins = [(a, b) for a in range(d_i[0], d_i[1] + 1) for b in range(d_j[0], d_j[1] + 1)]
    dec = {}
    dbatch = max(1, batch // 4)
    for b0 in range(0, len(dec_wins), dbatch):
        chunk = dec_wins[b0:b0 + dbatch]
        xs, xts = [], []
        for wi, wj in chunk:
            li, lj = wi * dlst - l_lo[0], wj * dlst - l_lo[1]
            lat = latents[:4, li:li + dls, lj:lj + dls]
            up = np.repeat(np.repeat(lat, LC, axis=1), LC, axis=2)
            noise = noise_patch(seed + 5819, wi * DEC_ST, wj * DEC_ST, DEC_S, DEC_S, 1, DEC_S, DEC_S)
            xt = sin_t * noise * SIGMA_DATA
            xts.append(xt)
            xs.append(np.concatenate([xt / SIGMA_DATA, up], axis=0))
        xts = np.stack(xts).astype(np.float32)
        pred = -models.run("decoder", {"x": np.stack(xs).astype(np.float32),
                                       "noise_labels": np.full(len(chunk), t_dec, dtype=np.float32)})
        new = (cos_t * xts - sin_t * SIGMA_DATA * pred) / SIGMA_DATA
        for k, w in enumerate(chunk):
            dec[w] = new[k].astype(np.float32)
    residual = _accumulate(dec, (pi1, pj1), (pi2, pj2), DEC_S, DEC_ST, WW_DEC, 1)[0]
    residual = residual * RESIDUAL_STD + RESIDUAL_MEAN
    del dec

    lowfreq = latents[4, pi1 // LC - l_lo[0]:pi2 // LC - l_lo[0], pj1 // LC - l_lo[1]:pj2 // LC - l_lo[1]]
    lowfreq = lowfreq * LOWFREQ_STD + LOWFREQ_MEAN

    p_h, p_w = residual.shape
    decoded = residual + bilinear_resize_extrapolated(lowfreq, p_h, p_w)
    new_low = gaussian_blur_clamped(bilinear_resize(decoded, p_h // LC, p_w // LC), sigma)
    elev_sqrt = residual + bilinear_resize(new_low, p_h, p_w)
    elev = np.sign(elev_sqrt) * elev_sqrt * elev_sqrt
    return elev[i1 - pi1:i2 - pi1, j1 - pj1:j2 - pj1].astype(np.float32)
