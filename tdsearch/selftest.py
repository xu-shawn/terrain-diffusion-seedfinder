"""Check the Python port against the mod's own Java code (the harness). CPU only.

Usage: python3 -m tdsearch.selftest [--elev]
Default: RNG, synthetic map and one coarse region (~20 s). --elev adds a 300x300 full-resolution
elevation region (about a minute per side on a desktop CPU).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

from . import pipeline as P
from .rng import noise_patch
from .synth import SyntheticMap

ROOT = os.path.join(os.path.dirname(__file__), "..")


def harness(*args):
    subprocess.run([os.path.join(ROOT, "harness", "run.sh"), *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    seed = 12345
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "a.bin")

        harness("noise", -77, -100, 30, 64, 64, 5, 64, f)
        d = np.abs(np.fromfile(f, dtype="<f4").reshape(5, 64, 64) - noise_patch(-77, -100, 30, 64, 64, 5, 64, 64)).max()
        print(f"gaussian noise   max |diff| {d:.3g}")
        ok &= d == 0

        harness("syn", seed, -40, 100, 24, 164, f)
        d = np.abs(np.fromfile(f, dtype="<f4").reshape(5, 64, 64) - SyntheticMap(seed).sample(-40, 100, 24, 164)).max()
        print(f"synthetic map    max |diff| {d:.3g}")
        ok &= d < 1e-4

        models = P.Models(providers=["CPUExecutionProvider"])
        harness("coarse", seed, 0, 0, 60, 60, f)
        j = np.fromfile(f, dtype="<f4").reshape(7, 60, 60)
        d = np.abs(j[:6] / j[6] - P.coarse_region(models, seed, 0, 0, 60, 60)).max(axis=(1, 2))
        print(f"coarse stage     max |diff| per channel {np.round(d, 5)}  (channel 0 is sqrt-metres)")
        ok &= d[0] < 1e-3

        if "--elev" in sys.argv:
            harness("elev", seed, 8000, 8200, 8300, 8500, f)
            j = np.fromfile(f, dtype="<f4").reshape(300, 300)
            d = np.abs(j - P.elev_region(models, seed, 8000, 8200, 8300, 8500))
            print(f"30 m elevation   max |diff| {d.max():.4f} m, mean {d.mean():.5f} m")
            ok &= d.max() < 0.05
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
