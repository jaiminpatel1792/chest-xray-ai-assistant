import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np

from src.data.dataset import CheXpertDataset, CheXpertConfig, DEFAULT_LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

BATCH_SIZE = 16
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(valid_loader, model):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            ys.append(y.numpy())
            ps.append(probs)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    # AUROC
    aucs = {}
    for i, label in enumerate(DEFAULT_LABELS):
        try:
            aucs[label] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            aucs[label] = None

    valid_aucs = [v for v in aucs.values() if v is not None]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    # Precision / Recall / F1
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    cls_metrics = {}
    for i, label in enumerate(DEFAULT_LABELS):
        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

        cls_metrics[label] = {
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1": float(round(f1, 4)),
        }

    return aucs, mean_auc, cls_metrics


def main():
    # Load validation dataset
    image_root = DATA_DIR / "raw/downloads/CheXpert-v1.0-small/CheXpert-v1.0-small"
    valid_csv = DATA_DIR / "processed/valid_5labels_fixed.csv"

    valid_ds = CheXpertDataset(
        CheXpertConfig(image_root=image_root, csv_path=valid_csv, labels=DEFAULT_LABELS),
        image_size=IMG_SIZE,
    )

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    ckpt = torch.load(MODELS_DIR / "best.pt", map_location=DEVICE)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(DEFAULT_LABELS))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)

    print("Loaded model from best.pt")
    print("Evaluating...")

    aucs, mean_auc, cls_metrics = evaluate(valid_loader, model)

    print("\nAUROC:")
    for k, v in aucs.items():
        if v is None:
            print(f"{k:18s} AUROC: N/A")
        else:
            print(f"{k:18s} AUROC: {v:.4f}")

    if mean_auc is not None:
        print(f"\nMean AUROC: {mean_auc:.4f}")
    else:
        print("\nMean AUROC: N/A")

    print("\nPrecision / Recall / F1:")
    for label, vals in cls_metrics.items():
        print(
            f"{label:18s} "
            f"Precision: {vals['precision']:.4f}  "
            f"Recall: {vals['recall']:.4f}  "
            f"F1: {vals['f1']:.4f}"
        )

    # Save metrics
    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "classification_metrics.json"

    payload = {
        "mean_auc": mean_auc,
        "per_label_auc": aucs,
        "classification_metrics": cls_metrics,
        "threshold": 0.5,
        "labels": DEFAULT_LABELS,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"\nSaved metrics ✅ {out_path}")


if __name__ == "__main__":
    main()
