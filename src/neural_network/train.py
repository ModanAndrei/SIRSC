"""Script pentru antrenarea YOLO v11 folosind datele din Etapele 3 și 4."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained COCO weights")
    args = parser.parse_args()

    # Alegem arhitectura: fie pornim de la zero (.yaml), fie folosim greutăți preantrenate (.pt)
    model_type = "yolo11n.pt" if args.pretrained else "yolo11n.yaml"
    model = YOLO(model_type)
    result = model.train(
        data=str(DATASET),
        epochs=max(10, args.epochs),
        batch=args.batch_size,
        imgsz=args.imgsz,
        fraction=args.fraction,
        patience=args.patience,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True, # Folosim cosine scheduler pentru o scădere mai lină a LR
        degrees=5.0, # Mici rotații pentru robustețe
        perspective=0.001,
        translate=0.1,
        scale=0.3,
        shear=0.1,
        fliplr=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0, # Combinăm mai multe imagini într-una singură
        mixup=0.1,
        pretrained=args.pretrained,
        workers=0,
        device="cpu",
        amp=False,
        project=str(ROOT / "runs"),
        name="gtsrb_15_classes_50ep",
        exist_ok=True,
        verbose=True,
    )
    # După ce terminăm, salvăm cel mai bun model în folderul centralizat
    run_dir = Path(result.save_dir)
    best = run_dir / "weights" / "best.pt"
    destination = ROOT / "models" / "trained_model.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)

    # Copiem rezultatele și în folderul principal de rezultate pentru a le avea la îndemână
    results_csv = run_dir / "results.csv"
    history_path = ROOT / "results" / "training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open(encoding="utf-8", newline="") as source, history_path.open("w", encoding="utf-8", newline="") as target:
        rows = list(csv.reader(source))
        csv.writer(target).writerows(rows)
    (ROOT / "results" / "hyperparameters.yaml").write_text(
        "model: %s\npretrained: %s\ndevice: cpu\namp: false\nepochs: %d\nbatch_size: %d\nimgsz: %d\nfraction: %.3f\npatience: %d\noptimizer: AdamW\nlearning_rate: 0.001\nscheduler: cosine\ndegrees: 5.0\nperspective: 0.001\nshear: 0.1\nlighting_augmentation: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4\nadvanced_aug: mosaic=1.0, mixup=0.1\n" % (model_type, str(args.pretrained).lower(), max(10, args.epochs), args.batch_size, args.imgsz, args.fraction, args.patience),
        encoding="utf-8",
    )
    print(f"Trained model saved to {destination}")


if __name__ == "__main__":
    main()