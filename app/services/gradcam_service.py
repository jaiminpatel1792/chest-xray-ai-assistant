from io import BytesIO
import base64

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from src.data.dataset import DEFAULT_LABELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


def generate_gradcam_from_model(model, x, target_class_idx):
    features = model.features(x)
    features = features.clone()
    features.requires_grad_(True)
    features.retain_grad()

    out = torch.relu(features)
    pooled = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
    pooled = torch.flatten(pooled, 1)
    logits = model.classifier(pooled)

    score = logits[0, target_class_idx]

    model.zero_grad()
    score.backward()

    grads = features.grad
    acts = features

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


def generate_gradcam_base64(image: Image.Image, model, probs):
    raw_img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    target_idx = int(np.argmax(np.array(probs)))

    img_np = np.array(raw_img).astype(np.float32) / 255.0
    x = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.to(DEVICE)

    cam, _ = generate_gradcam_from_model(model, x, target_idx)
    overlay = overlay_heatmap(raw_img, cam)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded