from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


@dataclass
class CheXpertConfig:
    image_root: Path
    csv_path: Path
    labels: List[str]


class CheXpertDataset(Dataset):
    def __init__(self, cfg: CheXpertConfig, image_size: int = 224):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv_path)

        # basic schema checks
        missing = [c for c in (["Path"] + cfg.labels) if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        # transforms: resize + normalize for ImageNet-pretrained backbones
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        rel_path = Path(row["Path"])
        img_path = self.cfg.image_root / rel_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # load image
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        # multi-label target (float tensor for BCEWithLogitsLoss later)
        y = torch.tensor([float(row[l]) for l in self.cfg.labels], dtype=torch.float32)

        return x, y
