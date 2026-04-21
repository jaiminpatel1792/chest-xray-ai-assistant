import pandas as pd
from pathlib import Path

RAW_META = Path("data/raw/chexpert/metadata")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

def process(split: str):
    df = pd.read_csv(RAW_META / f"{split}.csv")

    # keep only Path + chosen labels
    df = df[["Path"] + LABELS]

    # baseline uncertainty handling: -1 -> 0
    df[LABELS] = df[LABELS].replace(-1, 0)

    df[LABELS] = df[LABELS].fillna(0)

    out_path = OUT_DIR / f"{split}_5labels.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} | shape={df.shape}")

if __name__ == "__main__":
    process("train")
    process("valid")
