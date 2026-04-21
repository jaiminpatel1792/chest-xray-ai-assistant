import json
from pathlib import Path
import yaml
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.data.dataset import CheXpertConfig, CheXpertDataset, DEFAULT_LABELS

import numpy as np
from sklearn.metrics import roc_auc_score

# ------------------
# Paths / Settings
# ------------------
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CFG = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())

BATCH_SIZE = CFG["training"]["batch_size"]
IMG_SIZE = CFG["model"]["img_size"]
LR = CFG["training"]["lr"]
MAX_STEPS = CFG["training"]["max_steps"]
LOG_EVERY = CFG["training"]["log_every"]
NUM_WORKERS = CFG["training"]["num_workers"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_transform(train: bool):
    tfms = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        tfms.append(transforms.RandomHorizontalFlip())
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(tfms)


def build_components():
    # Data
    image_root = Path(CFG["data"]["image_root"])
    train_csv = Path(CFG["data"]["train_csv"])
    valid_csv = Path(CFG["data"]["valid_csv"])

    train_ds = CheXpertDataset(
        CheXpertConfig(image_root=image_root, csv_path=train_csv, labels=DEFAULT_LABELS),
        image_size=IMG_SIZE,
    )
    valid_ds = CheXpertDataset(
        CheXpertConfig(image_root=image_root, csv_path=valid_csv, labels=DEFAULT_LABELS),
        image_size=IMG_SIZE,
    )

    # Loaders (keep num_workers=0 for Windows stability)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, len(DEFAULT_LABELS))
    model = model.to(DEVICE)

    # Loss (weighted BCE)
    pos_weight = json.loads((ARTIFACTS / "pos_weight.json").read_text())
    pos_weight_tensor = torch.tensor(
        [pos_weight[l] for l in DEFAULT_LABELS],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    return train_loader, valid_loader, model, criterion, optimizer, pos_weight


def sanity_one_batch(train_loader, model, criterion, optimizer):
    print("Model ready")
    print("Device:", DEVICE)

    model.train()

    print("Loading one batch...")
    x, y = next(iter(train_loader))
    print("Batch loaded", x.shape, y.shape)

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    optimizer.zero_grad()

    print("Forward pass...")
    logits = model(x)

    print("Compute loss...")
    loss = criterion(logits, y)

    print("Backward pass...")
    loss.backward()

    print("Optimizer step...")
    optimizer.step()

    print("One-batch sanity check passed")
    print("Batch loss:", loss.detach().item())


def mini_train(train_loader, model, criterion, optimizer, max_steps=200, log_every=20):
    model.train()
    running = 0.0

    history = []

    for step, (x, y) in enumerate(train_loader, start=1):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        loss_val = loss.detach().item()
        running += loss_val

        history.append({"step": step, "loss": loss_val})

        if step % log_every == 0:
            avg = running / log_every
            print(f"step {step}/{max_steps}  avg_loss={avg:.4f}")
            running = 0.0

        if step >= max_steps:
            break

    return history

def evaluate_auc(valid_loader, model):
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    aucs = {}
    for i, label in enumerate(DEFAULT_LABELS):
        y_true = all_targets[:, i]
        y_score = all_probs[:, i]

        # If a label has only one class in valid, AUROC is undefined
        if len(np.unique(y_true)) < 2:
            aucs[label] = None
        else:
            aucs[label] = float(roc_auc_score(y_true, y_score))

    valid_aucs = [v for v in aucs.values() if v is not None]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    return aucs, mean_auc

def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    print("Saved metrics", path)

def main():
    # Build everything
    train_loader, valid_loader, model, criterion, optimizer, pos_weight = build_components()

    print("Train batches:", len(train_loader))
    print("Valid batches:", len(valid_loader))
    print("pos_weight:", [round(pos_weight[l], 4) for l in DEFAULT_LABELS])
    print("Loss:", criterion.__class__.__name__)

    # 1-batch sanity check
    sanity_one_batch(train_loader, model, criterion, optimizer)

    # Mini-train run (200 batches)
    history = mini_train(train_loader, model, criterion, optimizer,
                     max_steps=MAX_STEPS, log_every=LOG_EVERY)
    
    # Save training history
    history_path = ARTIFACTS / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(history)

    print("Saved training history ✅", history_path)

    # Evaluate AUROC on validation set
    aucs, mean_auc = evaluate_auc(valid_loader, model)

    print("\nValidation AUROC:")
    for k in DEFAULT_LABELS:
        v = aucs[k]
        if v is None:
            print(f"{k:16s} AUROC: N/A (only one class in valid)")
        else:
            print(f"{k:16s} AUROC: {v:.4f}")

    if mean_auc is not None:
        print(f"Mean AUROC: {mean_auc:.4f}")
    else:
        print("Mean AUROC: N/A")

        # ---- Save metrics ----
    metrics = {
        "mean_auc": mean_auc,
        "per_label_auc": aucs,
        "labels": DEFAULT_LABELS,
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "lr": LR,
        "device": str(DEVICE),
    }
    save_metrics(metrics, ARTIFACTS / CFG["experiment"]["metrics_name"])

    # ---- Save BEST checkpoint by mean AUROC ----
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

    best_path = MODELS_DIR / CFG["experiment"]["checkpoint_name"]
    best_auc_path = ARTIFACTS / "best_auc.txt"

    # if best_auc file exists, read it; else start from -1
    if best_auc_path.exists():
        best_auc = float(best_auc_path.read_text().strip())
    else:
        best_auc = -1.0

    if mean_auc is not None and mean_auc > best_auc:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "labels": DEFAULT_LABELS,
                "img_size": IMG_SIZE,
                "mean_auc": mean_auc,
            },
            best_path,
        )
        best_auc_path.write_text(str(mean_auc))
        print(f"Saved BEST model  {best_path} (mean_auc improved {best_auc:.4f} -> {mean_auc:.4f})")
    else:
        print(f"Best model not updated (current mean_auc={mean_auc:.4f}, best={best_auc:.4f})")


if __name__ == "__main__":
    main()

