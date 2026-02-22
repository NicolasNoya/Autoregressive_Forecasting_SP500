"""
Reference LSTM baseline for the S&P 500 direction-forecasting challenge.

The ingestion program will call:

    model = get_model(train_loader)

where `train_loader` is a torch.utils.data.DataLoader that yields
(x, y) batches with:
    x : FloatTensor of shape (batch, WINDOW_SIZE, n_features)
    y : FloatTensor of shape (batch,)  — binary labels (1 = up, 0 = down)

`get_model` must return a trained torch.nn.Module whose forward pass accepts
a tensor of shape (batch, WINDOW_SIZE, n_features) and returns raw logits of
shape (batch,). The ingestion program applies sigmoid + 0.5 threshold itself.
"""

import torch
import torch.nn as nn


# ── Hyper-parameters (feel free to tune) ─────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
N_EPOCHS = 20
LEARNING_RATE = 1e-3
# ─────────────────────────────────────────────────────────────────────────────


class LSTMClassifier(nn.Module):
    """Sequence-to-one LSTM for binary direction prediction.

    Takes a window of shape (batch, seq_len, input_size) and returns
    a scalar logit per sample (shape: (batch,)).

    Architecture
    ------------
    LSTM (num_layers, hidden_size, dropout) → hidden state of last timestep
    → Linear(hidden_size → 1) → squeeze → logit
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last = out[:, -1, :]  # (batch, hidden_size)  — last timestep
        return self.head(last).squeeze(-1)  # (batch,)


def get_model(train_loader: torch.utils.data.DataLoader) -> nn.Module:
    """Train an LSTM on the provided DataLoader and return the trained model.

    Parameters
    ----------
    train_loader : DataLoader
        Yields (x, y) batches where x has shape (batch, WINDOW_SIZE, n_features)
        and y has shape (batch,) with values in {0, 1}.

    Returns
    -------
    model : nn.Module (in eval mode)
        Trained LSTMClassifier whose forward pass returns raw logits.
    """
    # Infer input size from the first batch
    x_sample, _ = next(iter(train_loader))
    input_size = x_sample.shape[-1]  # n_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = LSTMClassifier(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # (batch,)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1:>2}/{N_EPOCHS}  loss={avg_loss:.4f}")

    model.eval()
    return model
