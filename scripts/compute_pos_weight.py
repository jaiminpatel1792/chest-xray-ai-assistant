import json
from pathlib import Path

import pandas as pd

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

def main():
    df = pd.read_csv("data/processed/train_5labels_fixed.csv")
    n = len(df)

    pos_weight = {}
    for lab in LABELS:
        pos = float((df[lab] == 1).sum())
        neg = float(n - pos)
        # Avoid divide-by-zero just in case
        w = (neg / pos) if pos > 0 else 1.0
        pos_weight[lab] = w

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pos_weight.json"
    out_path.write_text(json.dumps(pos_weight, indent=2))

    print("Saved:", out_path)
    print("pos_weight values (neg/pos):")
    for k, v in pos_weight.items():
        print(f"{k:16s} {v:.4f}")

if __name__ == "__main__":
    main()
