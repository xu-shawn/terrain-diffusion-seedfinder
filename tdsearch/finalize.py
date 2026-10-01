"""Turn a Java-harness block dump for a finalist into coordinates, checks and preview images.

Usage: python3 -m tdsearch.finalize NAME SEED Z1 X1 Z2 X2 BLOCKS.bin CROP.npy CROP_I0 CROP_J0 [LAKE_Z LAKE_X]
The block dump comes from `harness/run.sh blocks SEED 6 Z1 X1 Z2 X2 BLOCKS.bin` (the mod's own code).
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .blocks import block_region
from .render import perspective, relief_image

SCALE = 6
SEA = 63
_S8 = np.ones((3, 3), dtype=bool)


def mc_yaw(dx, dz):
    """Minecraft yaw (degrees) that faces along (dx, dz): 0 = +Z (south), -90 = +X (east)."""
    return math.degrees(math.atan2(-dx, dz))


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    z1, x1, z2, x2 = (int(v) for v in sys.argv[3:7])
    blocks_path, crop_path = sys.argv[7], sys.argv[8]
    i0, j0 = int(sys.argv[9]), int(sys.argv[10])
    out_dir = os.path.join("results", name)
    os.makedirs(out_dir, exist_ok=True)

    raw = np.fromfile(blocks_path, dtype="<i2").reshape(2, z2 - z1, x2 - x1)
    Y = raw[0].astype(np.int32)  # the mod's surface value: blocks are solid below this Y
    info = dict(name=name, seed=seed, scale=SCALE, region=dict(z1=z1, x1=x1, z2=z2, x2=x2))

    # 1) Agreement between the mod's CPU result and the GPU-based port used for searching.
    crop = np.load(crop_path).astype(np.float32)
    try:
        port = block_region(crop, i0, j0, z1, x1, z2, x2, SCALE)
        d = port - Y
        info["gpu_vs_cpu"] = dict(exact=float((d == 0).mean()), within1=float((np.abs(d) <= 1).mean()),
                                  max_abs=int(np.abs(d).max()),
                                  water_agreement=float(((port < SEA) == (Y < SEA)).mean()))
    except Exception as exc:  # region not fully inside the crop
        info["gpu_vs_cpu"] = f"not comparable: {exc}"

    # 2) Lake: the water body nearest the expected position (or the largest enclosed one).
    water = Y < SEA
    lab, n = ndi.label(water, structure=_S8)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    sizes = ndi.sum(water, lab, np.arange(1, n + 1))
    if len(sys.argv) > 12:
        lz, lx = int(sys.argv[11]) - z1, int(sys.argv[12]) - x1
        k = int(lab[lz, lx])
        if k == 0:  # nearest water to the hint
            dist, idx = ndi.distance_transform_edt(~water, return_indices=True)
            k = int(lab[idx[0][lz, lx], idx[1][lz, lx]])
    else:
        cands = [(sizes[i - 1], i) for i in range(1, n + 1) if i not in border]
        k = max(cands)[1]
    lake = lab == k
    cz, cx = ndi.center_of_mass(lake)
    zs, xs = np.nonzero(lake)
    depth = SEA - Y[lake]
    info["lake"] = dict(center_x=int(x1 + cx), center_z=int(z1 + cz), area_blocks=int(lake.sum()),
                        area_km2=float(lake.sum() * 25 / 1e6), touches_region_edge=bool(k in border),
                        x_range=[int(x1 + xs.min()), int(x1 + xs.max())], z_range=[int(z1 + zs.min()), int(z1 + zs.max())],
                        water_depth_blocks_median=float(np.median(depth)), water_depth_blocks_max=int(depth.max()))

    # 3) Peak and steepness, measured in blocks from the lake shore.
    dist = ndi.distance_transform_edt(~lake)  # blocks
    pz, px = np.unravel_index(int(np.argmax(Y)), Y.shape)
    info["peak"] = dict(x=int(x1 + px), z=int(z1 + pz), y=int(Y.max()), height_above_lake_blocks=int(Y.max() - SEA),
                        elevation_m=int((Y.max() - SEA) * 5), blocks_from_lake=int(dist[pz, px]))
    info["rise_within"] = {f"{d}_blocks": int(Y[dist <= d].max() - SEA) for d in (200, 400, 600, 1000, 1600)}
    # Tallest drop inside any 200x200-block window (1 km): the "cliff" figure.
    ymax = ndi.maximum_filter(Y, size=200)
    ymin = ndi.minimum_filter(np.maximum(Y, SEA), size=200)
    rel = ymax - ymin
    rz, rx = np.unravel_index(int(np.argmax(rel)), rel.shape)
    info["steepest_200_block_window"] = dict(drop_blocks=int(rel.max()), x=int(x1 + rx), z=int(z1 + rz))

    # 4) Camera suggestion: on the far shore of the lake from the peak, looking at it.
    dz, dx = pz - cz, px - cx
    norm = math.hypot(dz, dx)
    uz, ux = dz / norm, dx / norm
    t = 0.0
    while True:  # walk away from the peak until we leave the lake
        zz, xx = int(cz - uz * t), int(cx - ux * t)
        if not (0 <= zz < Y.shape[0] and 0 <= xx < Y.shape[1]) or not lake[zz, xx]:
            break
        t += 4
    t += 40
    cam_z, cam_x = int(np.clip(cz - uz * t, 0, Y.shape[0] - 1)), int(np.clip(cx - ux * t, 0, Y.shape[1] - 1))
    cam_y = int(max(Y[cam_z, cam_x], SEA)) + 2
    cam_dist = math.hypot(pz - cam_z, px - cam_x)
    pitch = -math.degrees(math.atan2((Y.max() - cam_y) * 0.55, cam_dist))
    info["camera"] = dict(x=int(x1 + cam_x), y=cam_y, z=int(z1 + cam_z), yaw=round(mc_yaw(px - cam_x, pz - cam_z), 1),
                          pitch=round(pitch, 1))
    info["tp_command"] = (f"/tp @s {info['camera']['x']} {cam_y} {info['camera']['z']} "
                          f"{info['camera']['yaw']} {info['camera']['pitch']}")
    biomes, counts = np.unique(raw[1], return_counts=True)
    info["biome_ids"] = {int(b): int(c) for b, c in zip(biomes, counts)}

    # 5) Previews rendered from the mod's own block heights (5 m per block).
    metres = np.where(Y >= SEA, (Y - SEA) * 5.0, -20.0).astype(np.float32)
    step = max(1, Y.shape[0] // 1400)
    top = Image.fromarray(relief_image(metres[::step, ::step], res_m=5.0 * step))
    top.save(os.path.join(out_dir, "map_blocks.png"))
    Image.fromarray(relief_image(crop)).save(os.path.join(out_dir, "map_wide.png"))
    view = perspective(metres, (cam_z, cam_x), (pz, px), eye_above_m=10.0, res_m=5.0, fov_deg=80.0,
                       pitch_deg=-pitch * 0.9, max_dist_m=40000.0, haze_m=40000.0)
    view.save(os.path.join(out_dir, "view_from_lake.png"))
    # Aerial: further back and 250 blocks up.
    back = 500
    az, ax = int(np.clip(cam_z - uz * back, 0, Y.shape[0] - 1)), int(np.clip(cam_x - ux * back, 0, Y.shape[1] - 1))
    aerial = perspective(metres, (az, ax), (pz, px), eye_above_m=1500.0, res_m=5.0, fov_deg=75.0, pitch_deg=3.0,
                         max_dist_m=40000.0, haze_m=40000.0)
    aerial.save(os.path.join(out_dir, "view_aerial.png"))
    info["aerial_camera"] = dict(x=int(x1 + ax), y=int(max(Y[az, ax], SEA)) + 300, z=int(z1 + az))

    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
