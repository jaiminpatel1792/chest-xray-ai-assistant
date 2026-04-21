import pandas as pd

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

def main():
    df = pd.read_csv("data/processed/train_5labels_fixed.csv")

    print("Train rows:", len(df))
    print("\nPositive counts and rates:")
    for lab in LABELS:
        pos = int((df[lab] == 1).sum())
        rate = pos / len(df)
        print(f"{lab:16s}  pos={pos:7d}  rate={rate:.4f}")

if __name__ == "__main__":
    main()
