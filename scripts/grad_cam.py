from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms

from src.data.dataset import CheXpertConfig, CheXpertDataset, DEFAULT_LABELS
from src.utils.report_generator import generate_report


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

def disable_inplace_relu(module):
    for child_name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, child_name, nn.ReLU(inplace=False))
        else:
            disable_inplace_relu(child)

def get_model():
    ckpt = torch.load("models/best.pt", map_location=DEVICE)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(DEFAULT_LABELS))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    disable_inplace_relu(model)
    model.eval()
    return model


def get_dataset():
    image_root = Path("data/raw/downloads/CheXpert-v1.0-small/CheXpert-v1.0-small")
    valid_csv = Path("data/processed/valid_5labels_fixed.csv")
    ds = CheXpertDataset(
        CheXpertConfig(image_root=image_root, csv_path=valid_csv, labels=DEFAULT_LABELS),
        image_size=IMG_SIZE,
    )
    return ds


def get_raw_image_and_tensor(ds, idx=0):
    row = ds.df.iloc[idx]
    img_path = ds.cfg.image_root / Path(row["Path"])

    raw_img = Image.open(img_path).convert("RGB")
    raw_img = raw_img.resize((IMG_SIZE, IMG_SIZE))

    x, y = ds[idx]
    x = x.unsqueeze(0).to(DEVICE)

    return raw_img, x, y, row["Path"]


def generate_gradcam(model, x, target_class_idx):
    # Forward through feature extractor manually
    features = model.features(x)
    features = features.clone()              # avoid view/in-place issues
    features.requires_grad_(True)
    features.retain_grad()

    out = torch.relu(features)
    pooled = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
    pooled = torch.flatten(pooled, 1)
    logits = model.classifier(pooled)

    score = logits[0, target_class_idx]

    model.zero_grad()
    score.backward()

    grads = features.grad                    # [1, C, H, W]
    acts = features                          # [1, C, H, W]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    return cam, probs


def overlay_heatmap(raw_img, cam):
    img_np = np.array(raw_img).astype(np.float32) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def main():
    model = get_model()
    ds = get_dataset()

    sample_indices = [0, 10, 25, 50, 100]

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in sample_indices:
        raw_img, x, y, rel_path = get_raw_image_and_tensor(ds, idx=sample_idx)

        probs = torch.sigmoid(model(x)).detach().cpu().numpy()[0]
        target_idx = int(np.argmax(probs))
        target_label = DEFAULT_LABELS[target_idx]
        report_text = generate_report(
            DEFAULT_LABELS,
            probs.tolist(),
            ground_truth=y.numpy().tolist(),
            threshold=0.5,
        )

        cam, probs = generate_gradcam(model, x, target_idx)
        overlay = overlay_heatmap(raw_img, cam)

        safe_label = target_label.replace(" ", "_")
        out_path = out_dir / f"gradcam_idx{sample_idx}_{safe_label}.png"
        report_path = out_dir / f"gradcam_idx{sample_idx}_{safe_label}.txt"

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(raw_img)
        plt.axis("off")
        plt.title("Original")

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap="jet")
        plt.axis("off")
        plt.title("Grad-CAM")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"Overlay\n{target_label}: {probs[target_idx]:.3f}")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        report_path.write_text(report_text, encoding="utf-8")

        print(f"\nSaved Grad-CAM ✅ {out_path}")
        print(f"Saved Report ✅ {report_path}")

        print("\n" + "="*50)
        print("Image:", rel_path)

        print("\nGround Truth:")
        for label, val in zip(DEFAULT_LABELS, y.numpy()):
            print(f"{label:18s}: {int(val)}")

        print("\nPrediction:")
        print(f"Top label: {target_label} ({probs[target_idx]:.3f})")

        print("\nProbabilities:")
        for label, p in zip(DEFAULT_LABELS, probs):
            print(f"{label:18s}: {p:.4f}")

        print("\nGenerated Report:")
        print(report_text)

        # Check correctness
        gt = y.numpy()
        pred_binary = (probs > 0.5).astype(int)

        correct = (pred_binary == gt).all()

        print("\nResult:", "✅ Correct" if correct else "❌ Incorrect")


if __name__ == "__main__":
    main()