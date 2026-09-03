"""Create the required loss and validation-loss plot from Ultralytics history."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    history = pd.read_csv(ROOT / "results" / "training_history.csv")
    epochs = history["epoch"] + 1
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train/box_loss"], label="train box loss")
    plt.plot(epochs, history["val/box_loss"], label="validation box loss")
    plt.plot(epochs, history["train/cls_loss"], label="train class loss")
    plt.plot(epochs, history["val/cls_loss"], label="validation class loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Curbe de antrenare YOLO v11")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "docs" / "loss_curve.png", dpi=160)
    plt.close()
    print("Saved docs/loss_curve.png")


if __name__ == "__main__":
    main()