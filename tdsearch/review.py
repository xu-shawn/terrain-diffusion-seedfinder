"""Rank collected level-2 results and build contact sheets.

Usage: python3 -m tdsearch.review [--top 40] [--sheet out.png]
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from .analyze import analyze, lake_score
from .render import relief_image


def load_results(root="prod"):
    rows = []
    for path in glob.glob(os.path.join(root, "*", "results.jsonl")):
        host = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["host"] = host
                    rows.append(r)
    return rows


def crop_path(r, root="prod"):
    return os.path.join(root, r["host"], "crops", f"s{r['seed']}_{r['ci']}_{r['cj']}.npy")


def rank(rows, key=None):
    flat = []
    for r in rows:
        for l in r["lakes"]:
            flat.append((key(l) if key else l["score"], r, l))
    flat.sort(key=lambda t: -t[0])
    # one entry per site
    seen, out = set(), []
    for s, r, l in flat:
        k = (r["seed"], r["ci"], r["cj"])
        if k in seen:
            continue
        seen.add(k)
        out.append((s, r, l))
    return out


def contact_sheet(entries, path, thumb=420, cols=5):
    rows_n = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows_n * (thumb + 34)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for n, (s, r, l) in enumerate(entries):
        cp = crop_path(r)
        x, y = (n % cols) * thumb, (n // cols) * (thumb + 34)
        if os.path.exists(cp):
            el = np.load(cp).astype(np.float32)
            im = Image.fromarray(relief_image(el)).resize((thumb, thumb), Image.LANCZOS)
            sc = thumb / el.shape[0]
            dd = ImageDraw.Draw(im)
            dd.ellipse([l["lake_j"] * sc - 5, l["lake_i"] * sc - 5, l["lake_j"] * sc + 5, l["lake_i"] * sc + 5], outline=(255, 0, 0), width=2)
            dd.rectangle([l["view_j"] * sc - 4, l["view_i"] * sc - 4, l["view_j"] * sc + 4, l["view_i"] * sc + 4], outline=(255, 255, 0), width=2)
            sheet.paste(im, (x, y))
        d.text((x + 4, y + thumb + 2), f"#{n} s{r['seed']} ({r['ci']},{r['cj']}) sc{s:.0f} A{l['area_km2']:.1f}", fill=(255, 255, 255))
        d.text((x + 4, y + thumb + 17), f"h2 {l['h2']:.0f} h5 {l['h5']:.0f} h8 {l['h8']:.0f} riv {max(l['river_touch_m'], l['river_near_m']):.0f} {'enc' if l['enclosed'] else 'open'}", fill=(200, 200, 200))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--min-area", type=float, default=0.3)
    args = ap.parse_args()
    rows = load_results()
    print(len(rows), "sites refined;", sum(1 for r in rows if r["lakes"]), "with a lake")
    ranked = [t for t in rank(rows) if t[2]["area_km2"] >= args.min_area][:args.top]
    for n, (s, r, l) in enumerate(ranked):
        print(f"#{n:2d} {r['host']} seed {r['seed']:>8} c=({r['ci']:4d},{r['cj']:4d}) score {s:5.0f} view {l['view']:5.0f} "
              f"(elev {l['view_elev']:.0f} @ {l['view_dist'] / 1000:.1f}km) A {l['area_km2']:5.2f} {'enc ' if l['enclosed'] else 'open'} "
              f"h1 {l['h1']:4.0f} h2 {l['h2']:4.0f} h3 {l['h3']:4.0f} h5 {l['h5']:4.0f} h8 {l['h8']:4.0f} cliff {l['cliff5']:4.0f} "
              f"riv t{l['river_touch_m']:.0f}/n{l['river_near_m']:.0f} peak {r['peak']:.0f}")
    if args.sheet:
        contact_sheet(ranked, args.sheet)


if __name__ == "__main__":
    main()
