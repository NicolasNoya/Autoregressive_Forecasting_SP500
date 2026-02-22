import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Number of past trading days fed as a sequence to the model.
# Must be consistent between training and inference.
WINDOW_SIZE = 20

EVAL_SETS = ["test", "private_test"]


class SP500Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset for the S&P 500 direction-forecasting challenge.

    Each sample is a sliding window of shape (WINDOW_SIZE, n_features)
    ending at day `idx`. The target is the binary label of that last day
    (1 = close > prev_close, 0 otherwise).

    For the first WINDOW_SIZE-1 days, the window is left-padded with zeros.

    Parameters
    ----------
    features_path : Path
        Path to the features CSV (columns = feature names, rows = trading days
        in chronological order).
    labels_path : Path or None
        Path to the labels CSV (single column, same row order as features).
        Pass None for test sets where labels are withheld.
    window_size : int
        Number of past days (inclusive of the current day) in each sequence.
    """

    def __init__(
        self, features_path, labels_path=None, window_size=WINDOW_SIZE
    ):
        self.window_size = window_size
        self.X = pd.read_csv(features_path).values.astype(np.float32)
        self.n_features = self.X.shape[1]
        if labels_path is not None:
            self.y = pd.read_csv(labels_path).values.astype(np.float32).ravel()
        else:
            self.y = None  # test mode — labels are unknown

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Return (window, label) where window has shape (window_size, n_features).

        The label is the binary target for day `idx` (the last day of the window).
        During test mode (no labels), only the window tensor is returned.
        """
        window_start = max(0, idx - self.window_size + 1)
        window = self.X[window_start : idx + 1]  # (<=window_size, n_features)

        # Left-pad with zeros if we are at the beginning of the series
        if len(window) < self.window_size:
            padding = np.zeros(
                (self.window_size - len(window), self.n_features),
                dtype=np.float32,
            )
            window = np.concatenate([padding, window], axis=0)

        x = torch.tensor(
            window, dtype=torch.float32
        )  # (window_size, n_features)

        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)  # scalar
            return x, y
        return x  # test mode


def get_train_dataset(data_dir):
    """Build the training Dataset from separate features and labels CSVs."""
    data_dir = Path(data_dir)
    features_path = data_dir / "train" / "train_features.csv"
    labels_path = data_dir / "train" / "train_labels.csv"
    return SP500Dataset(features_path, labels_path)


def get_test_dataset(data_dir, eval_set):
    """Build a test Dataset (no labels) for a given evaluation split."""
    data_dir = Path(data_dir)
    features_path = data_dir / eval_set / f"{eval_set}_features.csv"
    return SP500Dataset(features_path, labels_path=None)


def evaluate_model(model, test_dataset):
    """Run inference over a test Dataset and return a DataFrame of 0/1 predictions.

    The model is expected to output raw logits (one scalar per sample).
    Predictions are thresholded at 0.5 after applying sigmoid.
    """
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            # test_dataset returns bare tensors (no label) — x is already the input
            x = x.to(device)
            logits = model(x)  # (batch,)
            probs = torch.sigmoid(logits)
            batch_preds = (probs >= 0.5).long().cpu().numpy().tolist()
            preds.extend(batch_preds)
    return pd.DataFrame({"Prediction": preds})


def main(data_dir, output_dir):
    from submission import get_model  # imported here so sys.path is set first

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # ── Training ──────────────────────────────────────────────────────────────
    train_dataset = get_train_dataset(data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    print("Training the model")
    start = time.time()
    model = get_model(train_loader)  # participant trains and returns the model
    train_time = time.time() - start

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("=" * 40)
    print("Evaluate the model")
    start = time.time()
    res = {}
    for eval_set in EVAL_SETS:
        test_dataset = get_test_dataset(data_dir, eval_set)
        res[eval_set] = evaluate_model(model, test_dataset)
    test_time = time.time() - start
    print(
        f"Completed Prediction. Total duration: {train_time + test_time:.1f}s"
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w+") as f:
        json.dump(dict(train_time=train_time, test_time=test_time), f)
    for eval_set in EVAL_SETS:
        filepath = output_dir / f"{eval_set}_predictions.csv"
        res[eval_set].to_csv(filepath, index=False)
    print()
    print("Ingestion Program finished. Moving on to scoring")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for codabench"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="",
    )

    args = parser.parse_args()
    sys.path.append(args.submission_dir)
    sys.path.append(Path(__file__).parent.resolve())

    main(Path(args.data_dir), Path(args.output_dir))
