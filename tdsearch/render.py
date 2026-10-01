"""Shaded-relief previews of elevation arrays."""

import numpy as np
from PIL import Image

_LAND_STOPS = [
    (0, (104, 144, 96)), (150, (132, 164, 104)), (500, (168, 176, 116)), (1000, (176, 158, 112)),
    (1800, (156, 130, 100)), (2800, (140, 124, 112)), (3800, (172, 168, 164)), (4800, (226, 226, 228)),
    (6500, (250, 250, 252)),
]


def _land_colors(elev):
    xs = np.array([s[0] for s in _LAND_STOPS], dtype=np.float32)
    out = np.empty(elev.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(elev, xs, [s[1][ch] for s in _LAND_STOPS])
    return out


def hillshade(elev, res_m, azimuth=315.0, altitude=40.0, z_factor=1.0):
    dy, dx = np.gradient(elev.astype(np.float32) * z_factor, res_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def relief_image(elev, res_m=30.0, water_level=0.0, z_factor=1.0):
    """RGB uint8 shaded relief. Water (elev < water_level) drawn flat blue, darker with depth."""
    land = np.maximum(elev, water_level)
    shade = hillshade(land, res_m, z_factor=z_factor)
    rgb = _land_colors(land - water_level) * (0.35 + 0.75 * shade[..., None])
    water = elev < water_level
    depth = np.clip((water_level - elev) / 400.0, 0, 1)
    wcol = np.stack([70 - 40 * depth, 130 - 60 * depth, 200 - 70 * depth], axis=-1)
    rgb[water] = wcol[water]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_relief(path, elev, res_m=30.0, max_side=1600, marks=None, **kw):
    img = Image.fromarray(relief_image(elev, res_m, **kw))
    if marks:
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        for (r, c, col) in marks:
            d.ellipse([c - 6, r - 6, c + 6, r + 6], outline=col, width=2)
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)
    img.save(path)
    return path


# ---------------------------------------------------------------------------------------------
# Perspective preview (heightmap ray-march, "voxel space" style)
# ---------------------------------------------------------------------------------------------

from numba import njit


@njit(cache=True)
def _march(height, color, cam_r, cam_c, cam_h, yaw, fov, pitch_px, focal, res_m, max_dist, out, sky_top, sky_bot, haze):
    img_h, img_w = out.shape[0], out.shape[1]
    n_r, n_c = height.shape
    for y in range(img_h):
        t = y / img_h
        for x in range(img_w):
            for k in range(3):
                out[y, x, k] = sky_top[k] * (1 - t) + sky_bot[k] * t
    horizon = img_h * 0.5 + pitch_px
    for x in range(img_w):
        ang = yaw + np.arctan((x - img_w * 0.5) / focal)
        cos_corr = np.cos(ang - yaw)
        dr, dc = -np.cos(ang), np.sin(ang)  # yaw 0 = north (towards -row), clockwise
        ybuf = img_h
        z = 30.0
        while z < max_dist and ybuf > 0:
            r = cam_r + dr * z / res_m
            c = cam_c + dc * z / res_m
            if r < 0 or c < 0 or r >= n_r - 1 or c >= n_c - 1:
                break
            r0, c0 = int(r), int(c)
            fr, fc = r - r0, c - c0
            h = (height[r0, c0] * (1 - fr) * (1 - fc) + height[r0, c0 + 1] * (1 - fr) * fc
                 + height[r0 + 1, c0] * fr * (1 - fc) + height[r0 + 1, c0 + 1] * fr * fc)
            zc = z * cos_corr
            sy = int(horizon - (h - cam_h) / zc * focal)
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
            z += max(6.0, z * 0.0025)


def perspective(elev, cam_rc, look_rc, eye_above_m=40.0, res_m=30.0, size=(1600, 900), fov_deg=70.0,
                max_dist_m=60000.0, pitch_deg=None, haze_m=45000.0, z_factor=1.0):
    """Render a view from cam_rc (row, col in pixels) towards look_rc. Returns PIL image."""
    land = np.maximum(elev, 0).astype(np.float32)
    color = relief_image(elev, res_m).astype(np.float32)
    w, h = size
    focal = (w * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    cam_r, cam_c = cam_rc
    dr, dc = look_rc[0] - cam_r, look_rc[1] - cam_c
    yaw = np.arctan2(dc, -dr)
    cam_h = float(land[int(cam_r), int(cam_c)]) + eye_above_m
    if pitch_deg is None:  # aim a third of the way up to the target
        dist = np.hypot(dr, dc) * res_m
        target_h = float(land[int(look_rc[0]), int(look_rc[1])])
        pitch_deg = np.degrees(np.arctan2((target_h - cam_h) * 0.45, dist))
    pitch_px = np.tan(np.radians(pitch_deg)) * focal
    out = np.zeros((h, w, 3), dtype=np.float32)
    _march(land * z_factor, color, float(cam_r), float(cam_c), cam_h * z_factor, float(yaw), np.radians(fov_deg),
           float(pitch_px), float(focal), float(res_m), float(max_dist_m), out,
           np.array([96.0, 150.0, 222.0]), np.array([206.0, 224.0, 240.0]), float(haze_m))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
