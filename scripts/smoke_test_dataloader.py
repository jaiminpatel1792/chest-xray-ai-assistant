from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CheXpertConfig, CheXpertDataset, DEFAULT_LABELS


def main():
    image_root = Path("data/raw/downloads/CheXpert-v1.0-small/CheXpert-v1.0-small")
    train_csv = Path("data/processed/train_5labels_fixed.csv")

    ds = CheXpertDataset(
        CheXpertConfig(image_root=image_root, csv_path=train_csv, labels=DEFAULT_LABELS),
        image_size=224,
    )

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,   # keep 0 on Windows for now
        pin_memory=False
    )

    x, y = next(iter(loader))
    print("batch x:", x.shape, x.dtype)
    print("batch y:", y.shape, y.dtype)
    print("y sample rows:", y[:3].tolist())


if __name__ == "__main__":
    main()
