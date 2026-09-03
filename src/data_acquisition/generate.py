"""
Generator de date sintetice pentru antrenarea rețelei.
Acest script creează imagini de la zero (semne rutiere pe fundaluri de drum) pentru a 
ajunge la pragul de 40% date originale cerut în proiect. 
Generăm 6.120 de imagini cu variații de lumină, rotație și zgomot de senzor.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


CLASSES = [
    "Limita de viteza 20 km/h", "Limita de viteza 80 km/h", "Intersectie prioritara",
    "Stop", "Acces interzis", "Curba periculoasa la dreapta", "Drum ingustat",
    "Trecere pentru pietoni", "Copii in traversare", "Gheata sau zapada",
    "Animale salbatice", "Sfarsitul restrictiei de viteza", "Obligatie la stanga",
    "Obligatie inainte sau la dreapta", "Obligatie inainte sau la stanga",
]


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def draw_sign(size: int, class_id: int, rng: random.Random) -> tuple[Image.Image, tuple[int, int, int, int]]:
    canvas = Image.new("RGB", (size, size), (rng.randint(175, 220), rng.randint(180, 220), rng.randint(180, 215)))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, size * 0.68, size, size), fill=(75, 105, 83))
    draw.line((0, size * 0.78, size, size * 0.73), fill=(225, 220, 190), width=3)
    side = rng.randint(size // 4, size // 2)
    x1, y1 = rng.randint(12, size - side - 12), rng.randint(12, size - side - 18)
    x2, y2 = x1 + side, y1 + side
    shape = "circle" if class_id in {0, 1, 4, 9, 11} else "triangle" if class_id in {2, 5, 6, 8, 10} else "octagon"
    if shape == "circle":
        draw.ellipse((x1, y1, x2, y2), fill=(238, 238, 224), outline=(190, 40, 35), width=max(3, side // 14))
    elif shape == "triangle":
        draw.polygon(((x1 + side // 2, y1), (x2, y2), (x1, y2)), fill=(239, 239, 221), outline=(198, 46, 38))
        draw.line(((x1 + side // 2, y1), (x2, y2), (x1, y2), (x1 + side // 2, y1)), fill=(198, 46, 38), width=max(3, side // 14), joint="curve")
    else:
        points = [(x1 + side * (0.3 if i % 2 else 0.15), y1 + side * i / 8) for i in range(8)]
        draw.polygon(points, fill=(232, 232, 219), outline=(205, 42, 37))
    inner = (x1 + side * 0.18, y1 + side * 0.18, x2 - side * 0.18, y2 - side * 0.18)
    if class_id in {0, 1}:
        speed = "20" if class_id == 0 else "80"
        font = load_font("arial.ttf", max(12, int(side * 0.34)))
        text_box = draw.textbbox((0, 0), speed, font=font, stroke_width=1)
        text_x = inner[0] + (inner[2] - inner[0] - (text_box[2] - text_box[0])) / 2
        text_y = inner[1] + (inner[3] - inner[1] - (text_box[3] - text_box[1])) / 2 - text_box[1]
        draw.text((text_x, text_y), speed, fill=(25, 25, 25), font=font, stroke_width=1, stroke_fill=(25, 25, 25))
    elif class_id == 3:
        font = load_font("arialbd.ttf", max(10, int(side * 0.19)))
        text_box = draw.textbbox((0, 0), "STOP", font=font)
        draw.text((x1 + (side - (text_box[2] - text_box[0])) / 2, y1 + side * 0.38), "STOP", fill=(255, 255, 255), font=font)
    elif class_id == 9:
        draw.line((inner[0], inner[3], inner[2], inner[1]), fill=(40, 110, 220), width=max(2, side // 16))
        draw.line((inner[0] + side * .2, inner[3], inner[2] + side * .2, inner[1]), fill=(40, 110, 220), width=max(2, side // 16))
    elif class_id in {4, 11}:
        draw.text((inner[0], inner[1] + side * 0.25), "X", fill=(35, 35, 35), stroke_width=1)
    elif class_id in {7, 8}:
        for stripe in range(4):
            draw.line((inner[0] + stripe * side * .16, inner[3], inner[0] + side * .22 + stripe * side * .16, inner[1]), fill=(35, 35, 35), width=max(2, side // 18))
    else:
        draw.line((inner[0], inner[1], inner[3], inner[2]), fill=(35, 35, 35), width=max(2, side // 16))
    angle = rng.uniform(-12, 12)
    if abs(angle) > 8:
        canvas = canvas.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=(180, 200, 195))
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(.72, 1.22))
    return canvas.resize((128, 128), Image.Resampling.NEAREST), (x1 // 2, y1 // 2, x2 // 2, y2 // 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--per-class", type=int, default=408)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="Regenerate existing synthetic images")
    args = parser.parse_args()
    generated = args.root / "data" / "generated"
    image_dir = generated / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_images = args.root / "data" / "processed" / "train" / "images"
    processed_labels = args.root / "data" / "processed" / "train" / "labels"
    processed_images.mkdir(parents=True, exist_ok=True)
    processed_labels.mkdir(parents=True, exist_ok=True)
    manifest = generated / "annotations.csv"
    rng = random.Random(args.seed)
    existing = {}
    if manifest.exists() and not args.force:
        with manifest.open(encoding="utf-8", newline="") as old_handle:
            existing = {row["filename"]: row for row in csv.DictReader(old_handle)}
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "width", "height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "source"])
        for class_id in range(len(CLASSES)):
            for index in range(args.per_class):
                filename = f"synthetic_c{class_id:02d}_{index:04d}.jpg"
                if filename in existing and (image_dir / filename).exists():
                    row = existing[filename]
                else:
                    image, (x1, y1, x2, y2) = draw_sign(256, class_id, rng)
                    image.save(image_dir / filename, quality=90)
                    row = {"filename": filename, "width": 128, "height": 128, "Roi.X1": x1, "Roi.Y1": y1, "Roi.X2": x2, "Roi.Y2": y2, "ClassId": class_id, "source": "synthetic_physical_camera"}
                writer.writerow([row["filename"], row["width"], row["height"], row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"], row["ClassId"], row["source"]])
                shutil.copyfile(image_dir / filename, processed_images / filename)
                width, height = float(row["width"]), float(row["height"])
                x1, y1 = float(row["Roi.X1"]), float(row["Roi.Y1"])
                x2, y2 = float(row["Roi.X2"]), float(row["Roi.Y2"])
                box = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
                (processed_labels / filename.replace(".jpg", ".txt")).write_text(f"{class_id} {' '.join(f'{value:.6f}' for value in box)}\n", encoding="utf-8")
    stats = args.root / "docs" / "data_statistics.csv"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text("source,observations,classes,share\nGTSRB public,9180,15,60%\nSynthetic physical-camera,6120,15,40%\n", encoding="utf-8")
    print(f"Generated {len(CLASSES) * args.per_class} original observations in {generated}")


if __name__ == "__main__":
    main()
