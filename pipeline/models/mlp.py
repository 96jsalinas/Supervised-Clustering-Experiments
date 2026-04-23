import numpy as np
import torch
import torch.nn as nn

from pipeline.base import BaseModel


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class _MLPNet(nn.Module):
    """Plain Linear + activation stack. No BatchNorm so LRP composites apply
    cleanly. Activation defaults to ReLU but can be swapped via the config.

    If mean/std buffers are provided the forward pass z-scores inputs
    before the first Linear layer so attributors that call `net(raw_X)`
    still receive correctly-scaled activations.
    """

    def __init__(self, in_features: int, hidden_sizes: list[int],
                 n_classes: int, dropout: float,
                 activation_cls: type = nn.ReLU,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None):
        super().__init__()
        if mean is not None:
            self.register_buffer("_mean",
                                 torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer("_std",
                                 torch.from_numpy(std.astype(np.float32)))
            self._standardize = True
        else:
            self._standardize = False

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._standardize:
            x = (x - self._mean) / self._std
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
        standardize       : bool        default False (z-score features using
                                        training-set stats; also applied at
                                        predict time and propagated to
                                        attribution via the transformed inputs)
        activation        : str         default "relu" — one of relu/gelu/tanh.
                                        BatchNorm is intentionally absent so
                                        any choice stays LRP-compatible.
        label_smoothing   : float       default 0.0 (passed through to
                                        nn.CrossEntropyLoss; non-zero values
                                        soften the target distribution which
                                        can preserve more intra-class signal
                                        in the logits than one-hot CE).
        n_classes         : int | None  default None  (inferred from y, so a
                                        training subsample missing a class
                                        will underestimate; set explicitly
                                        when the true class cardinality is
                                        known)
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
        standardize = bool(params.get("standardize", False))
        activation_name = str(params.get("activation", "relu")).lower()
        if activation_name not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation_name}'. "
                f"Supported: {sorted(_ACTIVATIONS)}"
            )
        activation_cls = _ACTIVATIONS[activation_name]
        label_smoothing = float(params.get("label_smoothing", 0.0))
        n_classes_cfg = params.get("n_classes", None)

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_classes = (
            int(n_classes_cfg) if n_classes_cfg is not None
            else max(int(y.max()) + 1, 2)
        )

        if standardize:
            mean = X.mean(axis=0).astype(np.float32)
            std = X.std(axis=0).astype(np.float32)
            std[std < 1e-8] = 1.0
        else:
            mean = None
            std = None

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

        net = _MLPNet(X.shape[1], hidden_sizes, n_classes, dropout,
                      activation_cls=activation_cls,
                      mean=mean, std=std).to(device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        epochs_since_best = 0

        n_tr = len(X_tr)
        for _epoch in range(epochs):
            net.train()
            perm = torch.randperm(n_tr)
            for start in range(0, n_tr, batch_size):
                b = perm[start:start + batch_size]
                optimizer.zero_grad()
                logits = net(X_tr[b])
                loss = loss_fn(logits, y_tr[b])
                loss.backward()
                optimizer.step()

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
