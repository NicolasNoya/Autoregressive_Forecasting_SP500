# Seed — Starter Template

Copy this file as `submission.py` and implement your model inside `get_model`.

The ingestion program will call `get_model(train_loader)` and expect back a trained
`torch.nn.Module` whose `forward(x)` returns probabilities in **[0, 1]**.

```python
import torch
import torch.nn as nn


def get_model(train_loader):
    """
    Train a model on the S&P 500 direction-forecasting task and return it.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Yields (x, y) batches where:
          x — FloatTensor of shape (batch, 20, n_features)
              A sliding window of the last 20 daily feature vectors.
              Features: Open, High, Low, Close, Volume (current and past days).
          y — FloatTensor of shape (batch,)
              Binary label: 1 if today's close > previous close, else 0.

    Returns
    -------
    model : torch.nn.Module
        Trained model in eval() mode.
        forward(x) must accept shape (batch, 20, n_features)
        and return probabilities in [0, 1] of shape (batch,).
        The ingestion program applies a 0.5 threshold to produce 0/1 predictions.
    """

    # --- Infer input size from the first batch ---
    x_sample, _ = next(iter(train_loader))
    input_size = x_sample.shape[-1]      # number of features per timestep
    seq_len    = x_sample.shape[1]       # window size (20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Define your model here ---
    # Example: single-layer LSTM + linear head + sigmoid
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size=64,
                                num_layers=1, batch_first=True)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x)           # (batch, seq_len, 64)
            last   = out[:, -1, :]          # (batch, 64) — last timestep
            return torch.sigmoid(self.head(last).squeeze(-1))  # (batch,)

    model     = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()              # BCELoss because sigmoid is already applied

    # --- Training loop ---
    N_EPOCHS = 10
    model.train()
    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            probs = model(x)              # (batch,)
            loss  = criterion(probs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{N_EPOCHS}  loss={total_loss/len(train_loader):.4f}")

    model.eval()
    return model
```

## Tips

- You can replace the LSTM with a GRU (`nn.GRU`), Transformer (`nn.TransformerEncoder`), or any other architecture.
- The window size is fixed at **20** timesteps by the ingestion program.
- Keep training time reasonable — the Codabench environment has limited CPU resources.
- You are free to add dropout, batch normalisation, learning rate schedulers, etc.
