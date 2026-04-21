import numpy as np
import torch
import torch.nn as nn

from pipeline.base import BaseModel


class _MLPNet(nn.Module):
    """Plain Linear + ReLU stack. No BatchNorm so LRP composites apply cleanly."""

    def __init__(self, in_features: int, hidden_sizes: list[int],
                 n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPModel(BaseModel):
    """Multi-layer perceptron classifier (PyTorch).

    Config keys (all under `params:` except `random_state`):
        hidden_sizes      : list[int]   default [64, 32]
        dropout           : float       default 0.0
        epochs            : int         default 200
        lr                : float       default 1e-3
        batch_size        : int         default 64
        weight_decay      : float       default 0.0
        val_fraction      : float       default 0.2   (0 disables early stopping)
        patience          : int         default 20    (epochs w/o val improvement)
        device            : str         default "cpu"
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        params = self.config.get("params", {})
        random_state = self.config.get("random_state", None)

        hidden_sizes = list(params.get("hidden_sizes", [64, 32]))
        dropout = float(params.get("dropout", 0.0))
        epochs = int(params.get("epochs", 200))
        lr = float(params.get("lr", 1e-3))
        batch_size = int(params.get("batch_size", 64))
        weight_decay = float(params.get("weight_decay", 0.0))
        val_fraction = float(params.get("val_fraction", 0.2))
        patience = int(params.get("patience", 20))
        device = params.get("device", "cpu")

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_classes = int(y.max()) + 1

        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        if val_fraction > 0:
            n_val = max(1, int(len(X) * val_fraction))
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
        else:
            tr_idx, val_idx = idx, np.array([], dtype=int)

        X_tr = torch.from_numpy(X[tr_idx]).to(device)
        y_tr = torch.from_numpy(y[tr_idx]).to(device)
        X_val = torch.from_numpy(X[val_idx]).to(device) if len(val_idx) else None
        y_val = torch.from_numpy(y[val_idx]).to(device) if len(val_idx) else None

        net = _MLPNet(X.shape[1], hidden_sizes, n_classes, dropout).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        epochs_since_best = 0

        n_tr = len(X_tr)
        for epoch in range(epochs):
            net.train()
            perm = torch.randperm(n_tr)
            for start in range(0, n_tr, batch_size):
                b = perm[start:start + batch_size]
                optim.zero_grad()
                logits = net(X_tr[b])
                loss = loss_fn(logits, y_tr[b])
                loss.backward()
                optim.step()

            if X_val is None:
                continue

            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(X_val), y_val).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self._model = net
        self._device = device
        self._n_features = X.shape[1]
        self._n_classes = n_classes

    @property
    def model(self) -> nn.Module:
        return self._model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
