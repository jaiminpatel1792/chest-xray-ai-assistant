from pathlib import Path
import pandas as pd

from src.data.dataset import CheXpertConfig, CheXpertDataset, DEFAULT_LABELS


def main():
    image_root = Path("data/raw/downloads/CheXpert-v1.0-small/CheXpert-v1.0-small")
    train_csv = Path("data/processed/train_5labels_fixed.csv")

    ds = CheXpertDataset(
        CheXpertConfig(image_root=image_root, csv_path=train_csv, labels=DEFAULT_LABELS),
        image_size=224,
    )

    x, y = ds[0]
    print("len:", len(ds))
    print("x shape:", x.shape, "dtype:", x.dtype)
    print("y:", y.tolist(), "shape:", y.shape, "dtype:", y.dtype)

    # also print the raw path for confidence
    df = pd.read_csv(train_csv)
    print("sample path:", df.loc[0, "Path"])


if __name__ == "__main__":
    main()
