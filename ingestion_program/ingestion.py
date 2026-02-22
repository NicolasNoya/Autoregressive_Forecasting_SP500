import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch


class SP500Dataset(torch.utils.data.Dataset):
    """A PyTorch Dataset class for the S&P 500 forecasting problem. It takes in a CSV file with features and target, and returns
    windows of features and targets for training a model. The window size can be specified, and if the window is larger than the
    specified index, it will be padded with zeros at the beggining.
    """

    def __init__(self, data_path, window_size=50):
        self.data_path = data_path
        self.window_size = window_size
        self.data = pd.read_csv(data_path)
        self.y = self.data["Target"].values
        self.X = self.data.drop(columns=["Target"]).values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """Return the features and target for the given index, the index will be the last day of the window,
        the final tensor should be of shape (window_size, n_features) and the target should be a tensor of shape window_size.
        """
        window_start = max(0, idx - self.window_size + 1)
        # if the window is smaller than the window size, we will pad it with zeros
        window = self.X[window_start : idx + 1]
        target = self.y[window_start : idx + 1]
        if len(window) < self.window_size:
            padding = self.window_size - len(window)
            window = torch.cat(
                [
                    torch.zeros((padding, self.X.shape[1])),
                    torch.tensor(window, dtype=torch.float32),
                ]
            )
            target = torch.cat(
                [
                    torch.zeros(padding, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32),
                ]
            )
        return window, target


EVAL_SETS = ["test", "private_test"]


def evaluate_model(model, X_test):
    """Evaluate the model on the test set. This function returns a  pandas DataFrame with the predictions for the test set."""
    y_pred = []
    test_loader = torch.utils.data.DataLoader(
        X_test, batch_size=1, shuffle=False
    )
    for x, _ in test_loader:
        y_pred.append(model(x)[-1])
    return pd.DataFrame({"Prediction": y_pred})


def get_dataset(data_dir):
    """Load the training dataset from the given data directory. This function returns a PyTorch Dataset object."""
    train_data_path = Path(data_dir / "train" / "train_features.csv")
    return SP500Dataset(train_data_path)


def main(data_dir, output_dir):
    # Here, you can import info from the submission module, to evaluate the
    # submission
    from submission import get_model

    X_train = get_dataset(data_dir)
    data_loader = torch.utils.data.DataLoader(
        X_train, batch_size=32, shuffle=True
    )

    print("Training the model")

    start = time.time()
    model = get_model(data_loader)
    train_time = time.time() - start

    print("=" * 40)
    print("Evaluate the model")
    start = time.time()
    res = {}
    for eval_set in EVAL_SETS:
        X_test = get_dataset(f"{eval_set}.csv")
        res[eval_set] = evaluate_model(model, X_test)
    test_time = time.time() - start
    print("-" * 10)
    duration = train_time + test_time
    print(f"Completed Prediction. Total duration: {duration}")

    # Write output files
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
