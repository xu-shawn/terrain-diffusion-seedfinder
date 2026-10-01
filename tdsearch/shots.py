"""Preview camera shots for a finalist site, coloured roughly like the in-game surface.

Builds one 5 m/block heightfield for the whole 38 km crop (30 m terrain upsampled the way the mod
does it), overlays the exact block heights and biomes from the Java dump where available, and
ray-marches each shot. Prints a /tp command per shot.

Usage: python3 -m tdsearch.shots SPEC.json   (see results/seed3375007/shots.json)
The spec names the 30 m crop (tdsearch.crop), the Java block dump (harness/run.sh blocks) and a list
of cameras: position, eye height above ground or absolute Y, look-at point, pitch and field of view.
"""

import json
import math
import os
import sys

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

from .pipeline import bilinear_resize

SCALE, SEA = 6, 63

# Rough in-game surface colours per mod biome id (tree cover folded in).
BIOME_RGB = {
    1: (120, 168, 84), 3: (236, 240, 244), 6: (88, 104, 58), 8: (70, 110, 52), 15: (58, 88, 60),
    16: (128, 148, 136), 19: (112, 122, 104), 31: (150, 170, 160), 32: (238, 242, 247),
    33: (214, 226, 240), 35: (128, 128, 126), 44: (58, 108, 214), 46: (58, 100, 200),
    108: (104, 140, 76), 115: (92, 120, 84), 116: (176, 190, 182),
}
WATER = (58, 108, 214)


def build_scene(spec):
    """Returns (height_m float32 [Z, X], colour uint8 [Z, X, 3], z_origin, x_origin) in blocks."""
    sun_az, sun_alt = spec.get("sun_azimuth", 105.0), spec.get("sun_altitude", 38.0)
    crop = np.load(spec["crop"]).astype(np.float32)
    n = crop.shape[0] * SCALE
    height = bilinear_resize(crop, n, n)  # block b <-> native (b + 0.5) / 6 - 0.5, as the mod does
    z0, x0 = spec["crop_i0"] * SCALE, spec["crop_j0"] * SCALE

    bz1, bx1, bz2, bx2 = spec["blocks_region"]
    raw = np.fromfile(spec["blocks"], dtype="<i2").reshape(2, bz2 - bz1, bx2 - bx1)
    y = raw[0].astype(np.float32)
    exact = np.where(y >= SEA, (y - SEA) * 5.0, -5.0)
    rz, rx = bz1 - z0, bx1 - x0
    height[rz:rz + exact.shape[0], rx:rx + exact.shape[1]] = exact

    land = np.maximum(height, 0.0)
    dz, dx = np.gradient(land, 5.0)
    slope = np.hypot(dx, dz)

    # Fallback colours outside the Java region: a simple elevation/slope rule mimicking the mod's classes.
    rgb = np.empty(height.shape + (3,), dtype=np.float32)
    rgb[:] = BIOME_RGB[15]
    rgb[(land < 40) & (slope < 0.08)] = BIOME_RGB[6]
    rgb[land > 1400] = BIOME_RGB[115]
    rgb[land > 2300] = BIOME_RGB[116]
    rgb[land > 2900] = BIOME_RGB[32]
    rgb[slope > 0.95] = BIOME_RGB[35]
    ids = raw[1]
    view = rgb[rz:rz + ids.shape[0], rx:rx + ids.shape[1]]
    for bid, col in BIOME_RGB.items():
        view[ids == bid] = col
    rgb[height < 0] = WATER

    # Sun shading (azimuth clockwise from north; rows run north -> south, so north is -row).
    az, alt = math.radians(sun_az), math.radians(sun_alt)
    sun = np.array([math.sin(az) * math.cos(alt), -math.cos(az) * math.cos(alt), math.sin(alt)])  # (x, z, up)
    norm = np.sqrt(dx * dx + dz * dz + 1.0)
    lambert = np.clip((-dx * sun[0] - dz * sun[1] + sun[2]) / norm, 0.0, 1.0)
    shade = 0.42 + 0.72 * lambert
    shade[height < 0] = 1.0
    rgb *= shade[..., None]
    return land, np.clip(rgb, 0, 255).astype(np.uint8), z0, x0


@njit(cache=True)
def _march(height, color, cam_r, cam_c, cam_h, yaw, pitch_px, focal, max_dist, out, haze):
    img_h, img_w = out.shape[0], out.shape[1]
    n_r, n_c = height.shape
    horizon = img_h * 0.5 + pitch_px
    sky_top = (86.0, 142.0, 226.0)
    sky_bot = (198.0, 220.0, 242.0)
    for y in range(img_h):
        t = min(1.0, max(0.0, 1.0 - (horizon - y) / (img_h * 0.9)))
        for x in range(img_w):
            for k in range(3):
                out[y, x, k] = sky_top[k] * (1 - t) + sky_bot[k] * t
    for x in range(img_w):
        ang = yaw + np.arctan((x - img_w * 0.5) / focal)
        cos_corr = np.cos(ang - yaw)
        dr, dc = -np.cos(ang), np.sin(ang)
        ybuf = img_h
        z = 3.0
        while z < max_dist and ybuf > 0:
            r = cam_r + dr * z
            c = cam_c + dc * z
            if r < 0 or c < 0 or r >= n_r - 1 or c >= n_c - 1:
                break
            r0, c0 = int(r), int(c)
            fr, fc = r - r0, c - c0
            h = (height[r0, c0] * (1 - fr) * (1 - fc) + height[r0, c0 + 1] * (1 - fr) * fc
                 + height[r0 + 1, c0] * fr * (1 - fc) + height[r0 + 1, c0 + 1] * fr * fc)
            sy = int(horizon - (h - cam_h) / (z * cos_corr) * focal)
            if sy < ybuf:
                f = 1.0 - np.exp(-z / haze)
                top = max(sy, 0)
                for k in range(3):
                    col = (color[r0, c0, k] * (1 - fr) * (1 - fc) + color[r0, c0 + 1, k] * (1 - fr) * fc
                           + color[r0 + 1, c0, k] * fr * (1 - fc) + color[r0 + 1, c0 + 1, k] * fr * fc)
                    col = col * (1 - f) + sky_bot[k] * f
                    for y in range(top, ybuf):
                        out[y, x, k] = col
                ybuf = top
            z += max(0.8, z * 0.002)


def mc_yaw(dx, dz):
    return math.degrees(math.atan2(-dx, dz))


def render(scene, cam_xz, eye_blocks, look_xz, pitch_up_deg, fov_deg=75.0, size=(1920, 1080), absolute_y=None):
    """Heights are handled in blocks (5 m). Returns (image, tp command, camera Y)."""
    land, color, z0, x0 = scene
    cam_r, cam_c = cam_xz[1] - z0, cam_xz[0] - x0
    ground_y = SEA + land[int(cam_r), int(cam_c)] / 5.0
    cam_y = absolute_y if absolute_y is not None else ground_y + eye_blocks
    w, h = size
    focal = (w * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    dx, dz = look_xz[0] - cam_xz[0], look_xz[1] - cam_xz[1]
    yaw_render = math.atan2(dx, -dz)
    out = np.zeros((h, w, 3), dtype=np.float32)
    _march(land / 5.0, color, float(cam_r), float(cam_c), float(cam_y - SEA), yaw_render,
           math.tan(math.radians(pitch_up_deg)) * focal, focal, 9000.0, out, 7000.0)
    tp = f"/tp @s {cam_xz[0]} {int(round(cam_y))} {cam_xz[1]} {mc_yaw(dx, dz):.1f} {-pitch_up_deg:.1f}"
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), tp, cam_y


def main():
    with open(sys.argv[1]) as f:
        spec = json.load(f)
    out_dir = spec.get("out", os.path.join(os.path.dirname(sys.argv[1]), "shots"))
    os.makedirs(out_dir, exist_ok=True)
    scene = build_scene(spec)
    land, _, z0, x0 = scene
    lines = []
    thumbs = []
    for shot in spec["shots"]:
        cam, look = tuple(shot["camera_xz"]), tuple(shot["look_at_xz"])
        img, tp, _ = render(scene, cam, shot.get("eye_above_ground", 0), look, shot["pitch_up"],
                            shot.get("fov", 75.0), absolute_y=shot.get("y"))
        img.save(os.path.join(out_dir, shot["name"] + ".png"))
        ground = SEA + land[cam[1] - z0, cam[0] - x0] / 5.0
        lines.append(f"{shot['name']}: {shot.get('note', '')}\n    {tp}    (ground there is Y {ground:.0f})")
        thumbs.append((shot["name"], img.resize((960, 540), Image.LANCZOS)))
    sheet = Image.new("RGB", (960 * 2, (540 + 30) * ((len(thumbs) + 1) // 2)), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for k, (name, im) in enumerate(thumbs):
        x, y = (k % 2) * 960, (k // 2) * 570
        sheet.paste(im, (x, y))
        d.text((x + 8, y + 544), name, fill=(235, 235, 235), font=font)
    sheet.save(os.path.join(out_dir, "contact_sheet.png"))
    with open(os.path.join(out_dir, "shots.txt"), "w") as f:
        f.write(f"Seed {spec['seed']}, World Scale {SCALE}. Previews are renders of the generated heights, not screenshots.\n\n")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
