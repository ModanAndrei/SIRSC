"""Pregătirea dataset-ului pentru YOLO folosind cele 15 clase alese din GTSRB."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image


# Cele 15 clase pe care le-am ales pentru proiect
SELECTED_CLASSES = [0, 5, 11, 14, 17, 20, 24, 27, 28, 30, 31, 32, 34, 36, 37]
# Denumirile lor în română pentru a fi afișate frumos în UI
CLASS_NAMES = {
    0: "Limita de viteza 20 km/h",
    5: "Limita de viteza 80 km/h",
    11: "Intersectie prioritara",
    14: "Stop",
    17: "Acces interzis",
    20: "Curba periculoasa la dreapta",
    24: "Drum ingustat",
    27: "Trecere pentru pietoni",
    28: "Copii in traversare",
    30: "Gheata sau zapada",
    31: "Animale salbatice",
    32: "Sfarsitul restrictiei de viteza",
    34: "Obligatie la stanga",
    36: "Obligatie inainte sau la dreapta",
    37: "Obligatie inainte sau la stanga",
}


def locate_training(root: Path) -> Path:
    candidates = [
        root / "GTSRB_Final_Training_Images" / "GTSRB" / "Final_Training" / "Images",
        root / "data" / "raw" / "GTSRB_Final_Training_Images" / "GTSRB" / "Final_Training" / "Images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Training images were not found in the workspace.")


def read_annotation(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle, delimiter=";"))


def yolo_box(annotation: dict[str, str]) -> tuple[float, float, float, float]:
    """Transformăm coordonatele din GTSRB în formatul cerut de YOLO (x_center, y_center, width, height) normalizate."""
    width = float(annotation["Width"])
    height = float(annotation["Height"])
    x1, y1 = float(annotation["Roi.X1"]), float(annotation["Roi.Y1"])
    x2, y2 = float(annotation["Roi.X2"]), float(annotation["Roi.Y2"])
    box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, box_width / width, box_height / height)


def convert_sample(image_path: Path, annotation: dict[str, str], output_image: Path, output_label: Path, class_id: int) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image.convert("RGB").save(output_image, format="JPEG", quality=95)
    box = " ".join(f"{value:.6f}" for value in yolo_box(annotation))
    output_label.write_text(f"{class_id} {box}\n", encoding="utf-8")


def collect_training_samples(training_root: Path) -> dict[int, list[tuple[Path, dict[str, str]]]]:
    samples: dict[int, list[tuple[Path, dict[str, str]]]] = {}
    for original_class in SELECTED_CLASSES:
        class_dir = training_root / f"{original_class:05d}"
        annotation_path = class_dir / f"GT-{original_class:05d}.csv"
        samples[original_class] = []
        with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
            for annotation in csv.DictReader(handle, delimiter=";"):
                samples[original_class].append((class_dir / annotation["Filename"], annotation))
    return samples


def split_samples(samples: list[tuple[Path, dict[str, str]]], seed: int) -> dict[str, list[tuple[Path, dict[str, str]]]]:
    """Împărțim datele în Train, Val și Test, având grijă să nu amestecăm imaginile din aceeași secvență (track)."""
    groups: dict[str, list[tuple[Path, dict[str, str]]]] = {}
    for sample in samples:
        track_id = sample[0].stem.split("_")[0]
        groups.setdefault(track_id, []).append(sample)
    shuffled_groups = list(groups.values())
    random.Random(seed).shuffle(shuffled_groups)
    group_count = len(shuffled_groups)
    train_end = max(1, round(group_count * 0.8))
    val_count = max(1, round(group_count * 0.1)) if group_count >= 3 else 0
    if train_end + val_count >= group_count and group_count >= 3:
        train_end = group_count - 2
        val_count = 1
    val_end = train_end + val_count
    return {
        "train": [item for group in shuffled_groups[:train_end] for item in group],
        "val": [item for group in shuffled_groups[train_end:val_end] for item in group],
        "test": [item for group in shuffled_groups[val_end:] for item in group],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    training_root = locate_training(args.root)
    output_root = args.root / "data" / "processed"
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for original_class, samples in collect_training_samples(training_root).items():
        class_id = SELECTED_CLASSES.index(original_class)
        split_samples_by_name = split_samples(samples, args.seed + original_class)
        for split, split_samples_list in split_samples_by_name.items():
            for image_path, annotation in split_samples_list:
                stem = f"c{original_class:02d}_{image_path.stem}"
                image_output = output_root / split / "images" / f"{stem}.jpg"
                label_output = output_root / split / "labels" / f"{stem}.txt"
                if image_output.exists() and label_output.exists():
                    continue
                convert_sample(
                    image_path,
                    annotation,
                    image_output,
                    label_output,
                    class_id,
                )
            counts[f"{split}:{original_class:02d}"] = len(split_samples_list)

    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "path: " + output_root.as_posix() + "\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"names: {json.dumps([CLASS_NAMES[value] for value in SELECTED_CLASSES], ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    (args.root / "config" / "selected_classes.json").write_text(
        json.dumps({"seed": args.seed, "original_to_yolo": {str(value): index for index, value in enumerate(SELECTED_CLASSES)}, "names": [CLASS_NAMES[value] for value in SELECTED_CLASSES]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.root / "data" / "processed" / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"Created {sum(counts.values())} YOLO samples in {output_root}")


if __name__ == "__main__":
    main()
