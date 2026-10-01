"""Level-2 scoring of a native-resolution elevation crop: lakes, feeder channels, peaks above them.

The mod only places water where elevation < 0, so a "lake" is an enclosed (or nearly enclosed)
below-sea-level basin and a "river" is a narrow below-sea-level channel next to it.
"""

import numpy as np
from scipy import ndimage as ndi

RES = 30.0
PX_KM2 = (RES / 1000.0) ** 2
_S8 = np.ones((3, 3), dtype=bool)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _border_labels(lab):
    return set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())


def find_water_features(elev, body_r=4, min_area_km2=0.2, max_area_km2=60.0):
    """Split water into wide bodies and narrow channels.

    Returns (water, lakes, channels). A lake is a wide body (at least 2*body_r+1 px across) that
    does not reach the crop border and is not ocean-sized. Channels are connected pieces of water
    narrower than that, with their extent in metres."""
    water = elev < 0
    wide = ndi.binary_opening(water, structure=_disk(body_r))
    # Regrow the opened core back into the water it came from so shores are not shaved off.
    wide = ndi.binary_dilation(wide, structure=_disk(2)) & water
    comp_lab, _ = ndi.label(water, structure=_S8)
    comp_border = _border_labels(comp_lab)

    body_lab, n_body = ndi.label(wide, structure=_S8)
    lakes = []
    if n_body:
        body_border = _border_labels(body_lab)
        areas = ndi.sum(wide, body_lab, np.arange(1, n_body + 1)) * PX_KM2
        slices = ndi.find_objects(body_lab)
        for k in range(1, n_body + 1):
            if k in body_border or not (min_area_km2 <= areas[k - 1] <= max_area_km2):
                continue
            mask = body_lab == k
            lakes.append(dict(label=k, mask=mask, area_km2=float(areas[k - 1]), slc=slices[k - 1],
                              enclosed=int(comp_lab[mask][0]) not in comp_border))

    thin = water & ~wide
    thin_lab, n_thin = ndi.label(thin, structure=_S8)
    channels = []
    if n_thin:
        sizes = ndi.sum(thin, thin_lab, np.arange(1, n_thin + 1))
        slices = ndi.find_objects(thin_lab)
        for k in range(1, n_thin + 1):
            if sizes[k - 1] < 12:
                continue
            sl = slices[k - 1]
            extent = float(np.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)) * RES
            if extent < 300.0:
                continue
            channels.append(dict(label=k, extent_m=extent, px=int(sizes[k - 1]), slc=sl))
    return water, lakes, channels, body_lab, thin_lab


def lake_metrics(elev, lake, channels, thin_lab, relief1k):
    mask = lake["mask"]
    dist = ndi.distance_transform_edt(~mask) * RES  # metres from the lake shore
    land = elev >= 0
    out = dict(area_km2=lake["area_km2"], enclosed=bool(lake["enclosed"]))
    ci, cj = ndi.center_of_mass(mask)
    out["lake_i"], out["lake_j"] = float(ci), float(cj)
    inside = elev[mask]
    out["depth_med"] = float(-np.median(inside))
    out["depth_max"] = float(-inside.min())
    for d_km in (1, 2, 3, 5, 8):
        sel = land & (dist <= d_km * 1000.0)
        out[f"h{d_km}"] = float(elev[sel].max()) if sel.any() else 0.0

    # "View" score: a tall point counts fully when it rises at >= 45 degrees from the shore,
    # and is discounted by sqrt(rise/run) when it stands further back.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(land & (dist >= 300.0), elev / dist, 0.0)
    view = elev * np.sqrt(np.clip(ratio, 0.0, 1.0))
    idx = int(np.argmax(view))
    out["view"] = float(view.ravel()[idx])
    out["view_i"], out["view_j"] = idx // elev.shape[1], idx % elev.shape[1]
    out["view_elev"] = float(elev.ravel()[idx])
    out["view_dist"] = float(dist.ravel()[idx])
    for h_min in (2000, 3000, 4000):
        sel = elev >= h_min
        out[f"tan{h_min}"] = float(ratio[sel].max()) if sel.any() else 0.0
    near = dist <= 5000.0
    out["cliff5"] = float(relief1k[near].max())

    # Channels touching the lake (river arms / outlets) and channels merely close to it.
    touch_zone = ndi.binary_dilation(mask, structure=_disk(2))
    near_zone = dist <= 1500.0
    touching = set(np.unique(thin_lab[touch_zone]).tolist()) - {0}
    nearby = set(np.unique(thin_lab[near_zone]).tolist()) - {0}
    ext = {c["label"]: c["extent_m"] for c in channels}
    out["river_touch_m"] = max([ext[k] for k in touching if k in ext] or [0.0])
    out["river_near_m"] = max([ext[k] for k in nearby if k in ext] or [0.0])
    out["n_river_touch"] = sum(1 for k in touching if k in ext and ext[k] >= 600.0)
    return out


def lake_score(m):
    river = max(m["river_touch_m"], 0.6 * m["river_near_m"])
    river_f = 1.0 + 0.3 * min(1.0, river / 3000.0)
    size_f = 0.75 + 0.25 * min(1.0, m["area_km2"] / 1.0)
    enc_f = 1.0 if m["enclosed"] else 0.9
    depth_f = 0.8 + 0.2 * min(1.0, m["depth_med"] / 5.0)
    return m["view"] * river_f * size_f * enc_f * depth_f


def analyze(elev):
    """Score a crop. Returns (summary dict, list of per-lake metric dicts sorted by score)."""
    elev = elev.astype(np.float32)
    water, lakes, channels, _, thin_lab = find_water_features(elev)
    land_elev = np.maximum(elev, 0)
    relief1k = ndi.maximum_filter(land_elev, size=67) - ndi.minimum_filter(land_elev, size=67)
    results = []
    for lake in lakes:
        m = lake_metrics(elev, lake, channels, thin_lab, relief1k)
        m["score"] = lake_score(m)
        results.append(m)
    results.sort(key=lambda m: -m["score"])
    summary = dict(peak=float(elev.max()), water_frac=float(water.mean()), n_lakes=len(lakes),
                   cliff_max=float(relief1k.max()),
                   max_channel_m=max([c["extent_m"] for c in channels] or [0.0]))
    return summary, results
