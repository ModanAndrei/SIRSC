"""Evaluate class predictions and create Etapa 5 evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "trained_model.pt")
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=96)
    args = parser.parse_args()
    model = YOLO(str(args.model))
    image_dir = ROOT / "data" / "processed" / "test" / "images"
    label_dir = ROOT / "data" / "processed" / "test" / "labels"
    expected: list[int] = []
    predicted: list[int] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_values = label_dir / f"{image_path.stem}.txt"
        first_label = label_values.read_text(encoding="utf-8").split()[0]
        expected.append(int(first_label))
        result = model.predict(str(image_path), conf=args.confidence, imgsz=args.imgsz, verbose=False)[0]
        predicted.append(int(result.boxes.cls[0]) if len(result.boxes) else -1)

    labels = list(range(15))
    accuracy = accuracy_score(expected, predicted)
    metrics = {
        "test_samples": len(expected),
        "test_accuracy": float(accuracy),
        "test_f1_macro": float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "test_precision_macro": float(precision_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "confidence_threshold": args.confidence,
        "imgsz": args.imgsz,
        "unmatched_predictions": predicted.count(-1),
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(expected, predicted, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Clasa prezisa")
    plt.ylabel("Clasa reala")
    plt.title("Matrice de confuzie - test GTSRB 15 clase")
    plt.tight_layout()
    plt.savefig(ROOT / "docs" / "confusion_matrix.png", dpi=160)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()