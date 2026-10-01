"""cubiomes-style flat biome map from a Java-harness block dump (which carries the mod's biome ids).

Usage: python3 -m tdsearch.biome_map BLOCKS.bin Z1 X1 Z2 X2 OUT.png [--mark NAME:X:Z ...]
One map pixel = 4 blocks (cubiomes' 1:4 biome scale), drawn at 2x. North is up.
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Mod biome id -> (name, colour). Colours are cubiomes' own (util.c initBiomeColors) for vanilla biomes.
# The three "*_sparse" biomes are the mod's own, so cubiomes has no colour for them; they get a
# lighter tint of their parent biome.
BIOMES = {
    1: ("plains", 0x8DB360), 3: ("snowy_plains", 0xFFFFFF), 5: ("desert", 0xFA9418), 6: ("swamp", 0x07F9B2),
    8: ("forest", 0x056621), 15: ("taiga", 0x0B6A5F), 16: ("snowy_taiga", 0x31554A), 17: ("savanna", 0xBDB25F),
    19: ("windswept_hills", 0x606060), 23: ("jungle", 0x507B0A), 26: ("badlands", 0xD94515),
    29: ("meadow", 0x60A445), 31: ("grove", 0x47726C), 32: ("snowy_slopes", 0xC4C4C4),
    33: ("frozen_peaks", 0xB0B3CE), 35: ("stony_peaks", 0x7B8F74), 41: ("warm_ocean", 0x0000AC),
    44: ("ocean", 0x000070), 46: ("cold_ocean", 0x202070), 48: ("frozen_ocean", 0x7070D6),
    108: ("forest_sparse (mod)", 0x4F9A5F), 115: ("taiga_sparse (mod)", 0x4E9C92),
    116: ("snowy_taiga_sparse (mod)", 0x6F9186),
}
STEP, ZOOM = 4, 2
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _rgb(c):
    return (c >> 16) & 255, (c >> 8) & 255, c & 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks")
    ap.add_argument("z1", type=int)
    ap.add_argument("x1", type=int)
    ap.add_argument("z2", type=int)
    ap.add_argument("x2", type=int)
    ap.add_argument("out")
    ap.add_argument("--mark", action="append", default=[], help="NAME:X:Z")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    raw = np.fromfile(args.blocks, dtype="<i2").reshape(2, args.z2 - args.z1, args.x2 - args.x1)
    ids = raw[1][::STEP, ::STEP]
    rgb = np.zeros(ids.shape + (3,), dtype=np.uint8)
    present = []
    for bid in np.unique(ids):
        name, col = BIOMES.get(int(bid), (f"id {bid}", 0xFF00FF))
        rgb[ids == bid] = _rgb(col)
        present.append((float((ids == bid).mean()), name, col))
    present.sort(reverse=True)
    tile = Image.fromarray(rgb).resize((ids.shape[1] * ZOOM, ids.shape[0] * ZOOM), Image.NEAREST)

    try:
        font = ImageFont.truetype(FONT_PATH, 15)
        small = ImageFont.truetype(FONT_PATH, 13)
        bold = ImageFont.truetype(FONT_PATH.replace("Sans.ttf", "Sans-Bold.ttf"), 17)
    except OSError:
        font = small = bold = ImageFont.load_default()

    left, top, legend_w, bottom = 78, 46, 290, 40
    w, h = tile.size
    img = Image.new("RGB", (left + w + legend_w, top + h + bottom), (24, 24, 28))
    img.paste(tile, (left, top))
    d = ImageDraw.Draw(img)
    d.text((left, 12), args.title, fill=(240, 240, 240), font=bold)

    def px(x, z):
        return left + (x - args.x1) / STEP * ZOOM, top + (z - args.z1) / STEP * ZOOM

    # Coordinate ticks every 512 blocks, with a faint grid like cubiomes-viewer.
    for x in range(-(-args.x1 // 512) * 512, args.x2, 512):
        cx, _ = px(x, args.z1)
        d.line([cx, top, cx, top + h], fill=(0, 0, 0), width=1)
        d.text((cx, top + h + 6), str(x), fill=(200, 200, 200), font=small, anchor="ma")
    for z in range(-(-args.z1 // 512) * 512, args.z2, 512):
        _, cz = px(args.x1, z)
        d.line([left, cz, left + w, cz], fill=(0, 0, 0), width=1)
        d.text((left - 8, cz), str(z), fill=(200, 200, 200), font=small, anchor="rm")
    d.text((left + w / 2, top + h + 24), "X (east →)", fill=(160, 160, 160), font=small, anchor="ma")
    d.text((8, top + h + 24), "Z (south ↓)", fill=(160, 160, 160), font=small)

    for spec in args.mark:
        name, x, z = spec.rsplit(":", 2)
        cx, cz = px(int(x), int(z))
        d.ellipse([cx - 7, cz - 7, cx + 7, cz + 7], outline=(0, 0, 0), width=4)
        d.ellipse([cx - 7, cz - 7, cx + 7, cz + 7], outline=(255, 255, 0), width=2)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((cx + 12 + dx, cz + dy), name, fill=(0, 0, 0), font=font, anchor="lm")
        d.text((cx + 12, cz), name, fill=(255, 255, 0), font=font, anchor="lm")

    lx, ly = left + w + 22, top + 4
    d.text((lx, ly), "Biomes (share of map)", fill=(240, 240, 240), font=font)
    ly += 30
    for frac, name, col in present:
        d.rectangle([lx, ly, lx + 22, ly + 16], fill=_rgb(col), outline=(90, 90, 90))
        d.text((lx + 32, ly + 8), f"{name}  {100 * frac:.1f}%", fill=(225, 225, 225), font=small, anchor="lm")
        ly += 25
    ly += 14
    for line in ("1 px = 2 blocks (sampled 1:4)", "grid every 512 blocks", "north is up"):
        d.text((lx, ly), line, fill=(150, 150, 150), font=small)
        ly += 20
    img.save(args.out)
    print(args.out, img.size, [(n, round(100 * f, 1)) for f, n, _ in present])


if __name__ == "__main__":
    main()
