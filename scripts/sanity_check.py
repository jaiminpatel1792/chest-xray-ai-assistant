from pathlib import Path
import yaml
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]

def main():
    cfg_path = ROOT / "configs" / "base.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    print("Repo root:", ROOT)
    print("Config loaded:", cfg["project"]["name"])
    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)

    # Check folders exist
    for k, rel in cfg["paths"].items():
        p = ROOT / rel
        print(f"{k}: {p} | exists={p.exists()}")

if __name__ == "__main__":
    main()
