import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import yaml

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_IMG_DIR = DATA_DIR / "raw" / "downloads" / "CheXpert-v1.0-small" / "CheXpert-v1.0-small"
LABELS_CSV = DATA_DIR / "processed" / "train_5labels_fixed.csv"
LABELS_YAML = ROOT / "configs" / "labels.yaml"

# Load labels
df = pd.read_csv(LABELS_CSV)

# Load label names
with open(LABELS_YAML, "r") as f:
    label_cfg = yaml.safe_load(f)

LABEL_NAMES = label_cfg["labels"]

# Pick one sample
row = df.iloc[0]
img_path = RAW_IMG_DIR / row["Path"]

# Load image
img = Image.open(img_path).convert("RGB")

# Show image
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis("off")
plt.title("Chest X-ray Sample")
plt.show()

# Show labels
print("Image path:", row["Path"])
print("\nLabels:")
for label in LABEL_NAMES:
    print(f"{label}: {int(row[label])}")
