import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)

PREFIX = "CheXpert-v1.0-small/"

def fix(split):
    df = pd.read_csv(DATA_DIR / f"{split}_5labels.csv")

    # remove dataset prefix from paths
    df["Path"] = df["Path"].str.replace(PREFIX, "", regex=False)

    out_path = OUT_DIR / f"{split}_5labels_fixed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    fix("train")
    fix("valid")
