from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def load_model(weights: str | Path) -> YOLO:
    return YOLO(str(weights))


def predict(model: YOLO, source, confidence: float = 0.35):
    return model.predict(source=source, conf=confidence, verbose=False)
