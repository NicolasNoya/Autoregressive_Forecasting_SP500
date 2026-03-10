# How to Participate

## Objective

Build a model that predicts whether the S&P 500 index will **close strictly above** the current day's close on the **next trading day**,
using only the provided historical OHLCV features.

## Input Features

Each sample in the dataset is a row in a CSV with the following columns (all values are for the **current trading day** or computed from past days only):

| Column | Description |
|--------|-------------|
| `Open` | Opening price of the trading day |
| `High` | Intraday high |
| `Low` | Intraday low |
| `Close` | Closing price of the trading day |
| `Volume` | Total trading volume |

The ingestion program constructs **sliding windows** of the last **50 trading days** for each sample and feeds them to your model as tensors of shape `(batch, 50, n_features)`.

## Target Label

- **1** — today's close will be **strictly above** the previous close
- **0** — today's close will be **at or below** the previous close

## What to Submit

Submit a single file named **`submission.py`** containing a function:

```python
def get_model(train_loader):
    ...
    return model
```

`train_loader` is a `torch.utils.data.DataLoader` yielding `(x, y)` batches where:
- `x` has shape `(batch, 50, n_features)` — a sliding window of the last 50 daily feature vectors
- `y` has shape `(batch,)` — binary labels `{0, 1}`

Your `get_model` function must **train the model** using the provided loader and return a trained `torch.nn.Module` whose `forward(x)` outputs **probabilities in [0, 1]** of shape `(batch,)` — i.e. sigmoid must already be applied inside `forward`.

See the **Seed** page for a working skeleton to get started.

## Evaluation Metric

Submissions are ranked by **ROC-AUC score** on the held-out test set.
A perfect model scores 1.0; random guessing scores ~0.5.

## How to Submit

1. Write your `submission.py` with a `get_model(train_loader)` function.
2. Zip it: `zip submission.zip submission.py`
3. Upload the zip on the **My Submissions** page.

## Rules

- Your model may only use information in the provided feature set — no external data sources.
- External Python libraries (e.g. `torch`, `sklearn`, `numpy`) are allowed.
- You may submit as many times as you like during the Development Phase.
- The private test set is only revealed after the phase ends.
