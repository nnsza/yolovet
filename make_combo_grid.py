# -*- coding: utf-8 -*-
"""Build publication-style 2x3 grid with row labels cow1 / cow2. Writes 组合发文图.png into 组合/."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DIR = Path(r"D:\my_software\rust_test\pre_train_software\组合")
OUT = DIR / "组合发文图.png"

CELL = 1080
LABEL_W = 220
PAD = 64
GAP = 36
BG = (248, 249, 252)
TEXT = (28, 35, 50)
RULE = (210, 216, 230)


def pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in (
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\msyhbd.ttf",
    ):
        p = Path(fp)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


def open_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def to_square_cell(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = im.crop((left, top, left + side, top + side))
    return cropped.resize((CELL, CELL), Image.Resampling.LANCZOS)


def main() -> None:
    if not DIR.is_dir():
        raise SystemExit(f"Directory not found: {DIR}")

    rows: list[tuple[str, list[str]]] = [
        ("cow1", ["1.png", "2.png", "3.png"]),
        ("cow2", ["4.png", "5.png", "6.png"]),
    ]

    font_label = pick_font(80)

    inner_w = LABEL_W + GAP + 3 * CELL + 2 * GAP
    inner_h = 2 * CELL + GAP
    W = inner_w + 2 * PAD
    H = inner_h + 2 * PAD

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    y_base = PAD
    for row_i, (label, names) in enumerate(rows):
        y = y_base + row_i * (CELL + GAP)

        lx0, ly0 = PAD, y
        bbox = draw.textbbox((0, 0), label, font=font_label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = lx0 + (LABEL_W - tw) // 2
        ty = ly0 + (CELL - th) // 2
        draw.text((tx, ty), label, fill=TEXT, font=font_label)
        draw.line(
            [(lx0 + LABEL_W - 1, ly0), (lx0 + LABEL_W - 1, ly0 + CELL)],
            fill=RULE,
            width=2,
        )

        x = PAD + LABEL_W + GAP
        for fn in names:
            p = DIR / fn
            if not p.is_file():
                raise SystemExit(f"missing: {p}")
            sq = to_square_cell(open_rgb(p))
            canvas.paste(sq, (x, y))
            draw.rectangle(
                [x, y, x + CELL - 1, y + CELL - 1],
                outline=RULE,
                width=3,
            )
            x += CELL + GAP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, "PNG", optimize=True)
    print(f"saved: {OUT}")
    print(f"size: {canvas.size[0]} x {canvas.size[1]} px")


if __name__ == "__main__":
    main()
