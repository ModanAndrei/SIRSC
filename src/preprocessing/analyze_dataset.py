"""Create compact EDA statistics for the selected processed YOLO dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    report: dict[str, object] = {"splits": {}, "missing_values": 0, "duplicate_files": 0, "invalid_labels": 0}
    hashes: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        images = sorted((processed / split / "images").glob("*.jpg"))
        labels = sorted((processed / split / "labels").glob("*.txt"))
        dimensions = Counter()
        class_counts: Counter[str] = Counter()
        for image_path in images:
            with Image.open(image_path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
            hashes[image_path.read_bytes().hex()] += 1
        for label_path in labels:
            lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                report["invalid_labels"] = int(report["invalid_labels"]) + 1
            for values in lines:
                if len(values) != 5 or not all(0 <= float(value) <= 1 for value in values[1:]):
                    report["invalid_labels"] = int(report["invalid_labels"]) + 1
                else:
                    class_counts[values[0]] += 1
        report["splits"][split] = {"images": len(images), "labels": len(labels), "class_counts": dict(class_counts), "dimensions": dict(dimensions)}
    report["duplicate_files"] = sum(count - 1 for count in hashes.values() if count > 1)
    output = processed / "eda_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EDA report written to {output}")


if __name__ == "__main__":
    main()