from pathlib import Path
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "chexpert" / "metadata"
CONFIG_DIR = ROOT / "configs"


def load_labels_config():
    cfg = yaml.safe_load((CONFIG_DIR / "labels.yaml").read_text())
    return cfg["labels"], cfg["baseline_uncertain_handling"]["strategy"]


def load_metadata(split: str):
    """
    split: train | valid | test
    """
    csv_path = DATA_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    labels, uncertain_strategy = load_labels_config()

    df = df[["Path"] + labels]

    if uncertain_strategy == "map_to_negative":
        df[labels] = df[labels].replace(-1, 0)

    return df


def summarize(df):
    print("Rows:", len(df))
    print("Label prevalence:")
    for col in df.columns[1:]:
        print(f"{col}: {df[col].mean():.4f}")


if __name__ == "__main__":
    print("Metadata loader ready.")
