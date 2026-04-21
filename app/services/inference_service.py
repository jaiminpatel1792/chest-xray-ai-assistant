from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.data.dataset import DEFAULT_LABELS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


class InferenceService:
    def __init__(self, model_path: str = "models/best.pt"):
        self.model_path = Path(model_path)
        self.labels = DEFAULT_LABELS
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self):
        ckpt = torch.load(self.model_path, map_location=DEVICE)

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(self.labels))
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()
        return model

    def preprocess(self, image: Image.Image):
        x = self.transform(image.convert("RGB"))
        x = x.unsqueeze(0).to(DEVICE)
        return x

    def predict(self, image: Image.Image):
        x = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        prob_map = {
            label: float(prob) for label, prob in zip(self.labels, probs)
        }

        top_idx = int(np.argmax(probs))
        top_label = self.labels[top_idx]
        top_probability = float(probs[top_idx])

        return {
            "top_label": top_label,
            "top_probability": top_probability,
            "probabilities": prob_map,
            "raw_probs": probs.tolist(),
        }