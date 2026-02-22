# Script to load the S&P500 data and create the splits for the benchmark
from pathlib import Path

import pandas as pd

PHASE = "dev_phase"

DATA_DIR = Path(PHASE) / "input_data"
REF_DIR = Path(PHASE) / "reference_data"

RAW_DATA_PATH = Path("raw_data") / "sp500_raw.csv"
TARGET_COL = "Target"


def make_csv(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)


if __name__ == "__main__":

    # Load the S&P500 data
    print(f"Loading data from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    # Separate features and target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    n = len(df)
    train_end = int(n * 0.6)
    test_end = int(n * 0.8)

    # Split chronologically: 60% train, 20% test, 20% private_test
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
    X_private_test, y_private_test = X.iloc[test_end:], y.iloc[test_end:]

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}, Samples: {n}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Store the data in the correct folders:
    # - input_data contains train data (both features and labels) and only
    #   test features so the test labels are kept secret
    # - reference_data contains the test labels for scoring
    for split, X_split, y_split in [
        ("train", X_train, y_train),
        ("test", X_test, y_test),
        ("private_test", X_private_test, y_private_test),
    ]:
        split_dir = DATA_DIR / split
        make_csv(X_split, split_dir / f"{split}_features.csv")
        label_dir = split_dir if split == "train" else REF_DIR
        make_csv(
            pd.DataFrame({TARGET_COL: y_split}),
            label_dir / f"{split}_labels.csv",
        )

    print("\nData splits created successfully!")
    print(
        f"{'Split':<15} {'Samples':<10} {'First Date':<15} {'Last Date':<15}"
    )
    print("-" * 55)
    for split, X_split in [
        ("train", X_train),
        ("test", X_test),
        ("private_test", X_private_test),
    ]:
        first_date = X_split["Date"].iloc[0]
        last_date = X_split["Date"].iloc[-1]
        print(
            f"{split:<15} {len(X_split):<10} {first_date:<15} {last_date:<15}"
        )
