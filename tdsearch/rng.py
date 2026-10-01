"""Port of the mod's PortableRng + GaussianNoisePatch (PCG 64/32 + Marsaglia polar)."""

import numpy as np
from numba import njit

_MULT = np.uint64(6364136223846793005)
_INC = np.uint64(1442695040888963407)
_GOLD = np.uint64(0x9E3779B9)
_M32 = np.uint64(0xFFFFFFFF)
_INV_2P32 = 1.0 / 4294967296.0


@njit(cache=True)
def tile_seed(base_seed, ty, tx):
    """PortableRng.tileSeed. base_seed is uint64; ty/tx are int32 (reinterpreted unsigned)."""
    h = base_seed * _GOLD
    h = h + (np.uint64(np.int64(ty)) & _M32)
    h = h * _GOLD + (np.uint64(np.int64(tx)) & _M32)
    return h


@njit(cache=True)
def fill_standard_normal(seed, out):
    state = seed
    n = out.size
    i = 0
    while i < n:
        state = state * _MULT + _INC
        x = (((state >> np.uint64(18)) ^ state) >> np.uint64(27)) & _M32
        rot = np.uint64(state >> np.uint64(59))
        r1 = ((x >> rot) | (x << ((np.uint64(32) - rot) & np.uint64(31)))) & _M32
        state = state * _MULT + _INC
        x = (((state >> np.uint64(18)) ^ state) >> np.uint64(27)) & _M32
        rot = np.uint64(state >> np.uint64(59))
        r2 = ((x >> rot) | (x << ((np.uint64(32) - rot) & np.uint64(31)))) & _M32
        v1 = 2.0 * (np.float64(r1) + 1.0) * _INV_2P32 - 1.0
        v2 = 2.0 * (np.float64(r2) + 1.0) * _INV_2P32 - 1.0
        s = v1 * v1 + v2 * v2
        if s > 0.0 and s < 1.0:
            f = np.sqrt(-2.0 * np.log(s) / s)
            out[i] = np.float32(v1 * f)
            i += 1
            if i < n:
                out[i] = np.float32(v2 * f)
                i += 1


@njit(cache=True)
def _noise_patch(base_seed, y0, x0, h, w, channels, tile_h, tile_w, out):
    ty0 = y0 // tile_h
    ty1 = (y0 + h - 1) // tile_h
    tx0 = x0 // tile_w
    tx1 = (x0 + w - 1) // tile_w
    tile_flat = np.empty(channels * tile_h * tile_w, dtype=np.float32)
    for ty in range(ty0, ty1 + 1):
        tile_y0 = ty * tile_h
        for tx in range(tx0, tx1 + 1):
            tile_x0 = tx * tile_w
            oy0 = max(y0, tile_y0)
            oy1 = min(y0 + h, tile_y0 + tile_h)
            ox0 = max(x0, tile_x0)
            ox1 = min(x0 + w, tile_x0 + tile_w)
            fill_standard_normal(tile_seed(base_seed, ty, tx), tile_flat)
            for c in range(channels):
                for py in range(oy0, oy1):
                    for px in range(ox0, ox1):
                        out[c, py - y0, px - x0] = tile_flat[
                            c * (tile_h * tile_w) + (py - tile_y0) * tile_w + (px - tile_x0)]


def noise_patch(base_seed, y0, x0, h, w, channels, tile_h, tile_w):
    """GaussianNoisePatch.generate. base_seed: python int (any sign), wrapped to 64 bits."""
    out = np.empty((channels, h, w), dtype=np.float32)
    _noise_patch(np.uint64(base_seed & 0xFFFFFFFFFFFFFFFF), y0, x0, h, w, channels, tile_h, tile_w, out)
    return out
