"""Port of the mod's SyntheticMapFactory: Perlin FBm quantile-matched to real-world tables."""

import json
import os

import numpy as np
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, FractalType, NoiseType

MODELS_DIR = os.environ.get("TD_MODELS", os.path.join(os.path.dirname(__file__), "..", "models"))

N_CHANNELS = 5
OCTAVES = (4, 2, 4, 4, 4)
BASE_FREQUENCY = 0.05

with open(os.path.join(MODELS_DIR, "pipeline_data.json")) as _f:
    _DATA = json.load(_f)
DATA_Q = np.asarray(_DATA["data_quantile_tables"], dtype=np.float32)
A_TEMP_STD = np.float32(_DATA["a_temp_std"])
B_TEMP_STD = np.float32(_DATA["b_temp_std"])
TEMP_STD_P1 = np.float32(_DATA["temp_std_p1"])
TEMP_STD_P99 = np.float32(_DATA["temp_std_p99"])

_f32 = np.float32


def make_fnl(seed, freq, octaves, lacunarity=2.0, gain=0.5):
    fnl = FastNoiseLite(seed=seed)
    fnl.noise_type = NoiseType.NoiseType_Perlin
    fnl.frequency = freq
    fnl.fractal_type = FractalType.FractalType_FBm
    fnl.fractal_octaves = octaves
    fnl.fractal_lacunarity = lacunarity
    fnl.fractal_gain = gain
    return fnl


_QX, _QY = np.meshgrid(np.arange(1024, dtype=np.float32) * 32, np.arange(1024, dtype=np.float32) * 32)
_Q_COORDS = np.stack([_QX.ravel(), _QY.ravel()]).astype(np.float32)


def build_noise_quantiles(fnl, n_quantiles=64, eps=1e-4):
    values = np.sort(fnl.gen_from_coords(_Q_COORDS).astype(np.float32))
    n = values.size
    eps = _f32(eps)
    q = np.empty(n_quantiles, dtype=np.float32)
    for i in range(n_quantiles):
        pct = _f32(eps + _f32(i) * _f32(_f32(1.0) - _f32(2) * eps) / _f32(n_quantiles - 1))
        idx = _f32(pct * _f32(n - 1))
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        q[i] = values[lo] + _f32(idx - _f32(lo)) * (values[hi] - values[lo])
    diffs = np.diff(q)
    pos = diffs[diffs > 0]
    min_diff = pos.min() if pos.size else _f32(1e-10)
    for i in range(1, n_quantiles):
        if q[i] <= q[i - 1]:
            q[i] = q[i - 1] + min_diff * _f32(0.1)
    return q


def interp(x, xp, fp):
    """SyntheticMapFactory.interp, vectorised in float32."""
    x = x.astype(np.float32)
    hi = np.clip(np.searchsorted(xp, x, side="right"), 1, xp.size - 1)
    lo = hi - 1
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    out = fp[lo] + t * (fp[hi] - fp[lo])
    out = np.where(x <= xp[0], fp[0], out)
    out = np.where(x >= xp[-1], fp[-1], out)
    return out.astype(np.float32)


class SyntheticMap:
    def __init__(self, world_seed):
        self.noises = []
        self.noise_q = []
        for ch in range(N_CHANNELS):
            fnl = make_fnl(int((world_seed + ch + 1) & 0x7FFFFFFF), BASE_FREQUENCY, OCTAVES[ch])
            self.noises.append(fnl)
            self.noise_q.append(build_noise_quantiles(fnl))

    def raw(self, ch, xs, ys):
        """Quantile-transformed channel at float32 coordinate arrays (x = column j, y = row i)."""
        coords = np.stack([np.ravel(xs), np.ravel(ys)]).astype(np.float32)
        vals = self.noises[ch].gen_from_coords(coords).astype(np.float32)
        return interp(vals, self.noise_q[ch], DATA_Q[ch]).reshape(np.shape(xs))

    def sample(self, x1, y1, x2, y2):
        """SyntheticMapFactory.sample: (5, y2-y1, x2-x1) = [elev_sqrt, temp, temp_std, precip, precip_std]."""
        xx, yy = np.meshgrid(np.arange(x1, x2, dtype=np.float32), np.arange(y1, y2, dtype=np.float32))
        elev, temp, temp_std, precip, precip_std = (self.raw(ch, xx, yy) for ch in range(N_CHANNELS))

        lapse = np.clip(_f32(-6.5) + _f32(0.0015) * precip, _f32(-9.8), _f32(-4.0)) / _f32(1000.0)
        temp = np.clip(temp + lapse * np.maximum(_f32(0), elev), _f32(-10), _f32(40))

        baseline = A_TEMP_STD * temp + B_TEMP_STD
        t01 = (temp_std - TEMP_STD_P1) / (TEMP_STD_P99 - TEMP_STD_P1)
        baseline_clipped = np.maximum(TEMP_STD_P1, -baseline)
        temp_std = t01 * (TEMP_STD_P99 - baseline_clipped) + baseline_clipped + baseline
        temp_std = np.maximum(temp_std, _f32(20))

        precip_std = precip_std * np.maximum(_f32(0), (_f32(185) - _f32(0.04111) * precip) / _f32(185))
        elev_sqrt = np.sign(elev) * np.sqrt(np.abs(elev))
        return np.stack([elev_sqrt, temp, temp_std, precip, precip_std]).astype(np.float32)
