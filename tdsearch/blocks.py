"""Port of the mod's LocalTerrainProvider.handleUpsampled + HeightConverter: native elevation -> block Y."""

import math

import numpy as np

from .pipeline import bilinear_resize
from .synth import make_fnl

NATIVE_RES = 30.0
SEA_LEVEL = 63
_NOISE_COARSE = make_fnl(99999, 1.0 / 24.0, 3, 2.0, 0.5)
_NOISE_FINE = make_fnl(88888, 1.0 / 6.0, 2, 2.0, 0.6)
f32 = np.float32


def _sobel_grad(padded):
    p = padded
    dx = (-p[:-2, :-2] + p[:-2, 2:] - 2 * p[1:-1, :-2] + 2 * p[1:-1, 2:] - p[2:, :-2] + p[2:, 2:]) / f32(8)
    dy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:] + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) / f32(8)
    return np.sqrt(dx * dx + dy * dy)


def tile_heights(elev_fn, z1, x1, scale, tile=256):
    """Block-surface elevation (metres, floored int) for one tile, as the mod computes it.

    elev_fn(i1, j1, i2, j2) must return native elevation for that native-pixel box."""
    z2, x2 = z1 + tile, x1 + tile
    i1n, j1n = z1 // scale, x1 // scale
    i2n, j2n = -((-z2) // scale), -((-x2) // scale)
    i1p, j1p, i2p, j2p = i1n - 2, j1n - 2, i2n + 2, j2n + 2
    n_h, n_w = i2p - i1p, j2p - j1p
    up = bilinear_resize(elev_fn(i1p, j1p, i2p, j2p), n_h * scale, n_w * scale)
    ci, cj = 2 * scale + (z1 - i1n * scale), 2 * scale + (x1 - j1n * scale)
    smooth = up[ci:ci + tile, cj:cj + tile]
    padded = up[ci - 1:ci + tile + 1, cj - 1:cj + tile + 1]
    px = f32(NATIVE_RES / scale)
    grad = _sobel_grad(padded)
    sf = np.minimum(f32(1), grad / f32(40.0 * px / NATIVE_RES))
    sf = sf * sf * np.sqrt(sf)
    xx, zz = np.meshgrid(np.arange(x1, x2, dtype=np.float32), np.arange(z1, z2, dtype=np.float32))
    coords = np.stack([xx.ravel(), zz.ravel()]).astype(np.float32)
    nc = _NOISE_COARSE.gen_from_coords(coords).reshape(tile, tile)
    nf = _NOISE_FINE.gen_from_coords(coords).reshape(tile, tile)
    out = smooth + nc * f32(100.0 * px / NATIVE_RES) * sf + nf * f32(70.0 * px / NATIVE_RES) * sf
    out = np.where(smooth < 0, smooth, out)
    return np.floor(out).astype(np.int32)


def to_block_y(meters, scale):
    """HeightConverter.convertToMinecraftHeight (value compared against y: solid where y < value)."""
    m = meters.astype(np.float64)
    res = NATIVE_RES / scale
    pos = np.trunc(np.float32(m) / np.float32(res)).astype(np.int32)
    neg = np.trunc(-np.sqrt(np.abs(m) + 10) + math.sqrt(10.0)).astype(np.int32) - 1
    return np.where(m >= 0, pos, neg) + SEA_LEVEL


def block_region(elev, i_org, j_org, z1, x1, z2, x2, scale):
    """Block Y map over [z1,z2) x [x1,x2) (multiples of 256) from a native elevation array with origin (i_org, j_org)."""
    def elev_fn(i1, j1, i2, j2):
        return elev[i1 - i_org:i2 - i_org, j1 - j_org:j2 - j_org]
    out = np.zeros((z2 - z1, x2 - x1), dtype=np.int32)
    for tz in range(z1, z2, 256):
        for tx in range(x1, x2, 256):
            out[tz - z1:tz - z1 + 256, tx - x1:tx - x1 + 256] = to_block_y(tile_heights(elev_fn, tz, tx, scale), scale)
    return out
