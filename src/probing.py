"""Scalar-mixing edge-probing utilities for Experiment 2C.

The implementation is adapted from the original ``Train_Edge_Probe.ipynb``
notebook.  A learnable softmax distribution mixes target-NP representations
across transformer layers, and a two-layer MLP predicts semantic specificity.

The 80/20 split is intentionally a *training/validation* split.  The held-out
20% of the independent SPECIFICITY dataset is used to validate the probe and
select a decision threshold.  SCOPEX is not used for probe optimization; the
trained probe is applied to SCOPEX only at downstream inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn


class ScalarMixingEdgeProbe(nn.Module):
    """Two-layer specificity probe with learnable scalar layer mixing.

    Parameters
    ----------
    num_layers:
        Number of hidden-state representations supplied per example.  This
        includes the embedding output if it is present in the extracted
        features.
    hidden_dim:
        Hidden-state dimensionality of the underlying language model.
    probe_hidden:
        Width of the probe's hidden layer.  The original experiment used 256.
    dropout:
        Dropout probability between the ReLU and output layer.  The original
        experiment used 0.1.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        probe_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.probe_hidden = int(probe_hidden)
        self.dropout = float(dropout)

        # Preserve the original initialization: all scalar parameters start at 1.
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))
        self.probe = nn.Sequential(
            nn.Linear(self.hidden_dim, self.probe_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.probe_hidden, 1),
        )

    def forward(self, layerwise_reprs: torch.Tensor):
        """Return specificity logits and normalized scalar-mixing weights.

        ``layerwise_reprs`` must have shape
        ``[batch_size, num_layers, hidden_dim]``.
        """
        if layerwise_reprs.ndim != 3:
            raise ValueError(
                "Expected layerwise_reprs with shape [batch, layers, hidden], "
                f"got {tuple(layerwise_reprs.shape)}."
            )
        if layerwise_reprs.shape[1] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {layerwise_reprs.shape[1]}."
            )
        if layerwise_reprs.shape[2] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {layerwise_reprs.shape[2]}."
            )

        mix_weights = torch.softmax(self.layer_weights, dim=0)
        mixed_repr = torch.sum(
            mix_weights.view(1, -1, 1) * layerwise_reprs,
            dim=1,
        )
        logits = self.probe(mixed_repr).squeeze(-1)
        return logits, mix_weights


# Backward-compatible alias matching the original notebook class name.
EdgeProbingMLP = ScalarMixingEdgeProbe


def load_layerwise_features(feature_directory: str | Path):
    """Load ``layer_*/features.npy`` files and stack them as ``[N, L, H]``.

    Labels are read from the first layer's ``labels.npy``.  The function also
    verifies that all layers contain the same number of examples and feature
    dimensionality.
    """
    feature_path = Path(feature_directory)
    layer_directories = sorted(
        (path for path in feature_path.glob("layer_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not layer_directories:
        raise FileNotFoundError(f"No layer_* directories found under {feature_path}.")

    all_features: list[np.ndarray] = []
    labels: np.ndarray | None = None
    expected_shape: tuple[int, int] | None = None

    for layer_directory in layer_directories:
        feature_file = layer_directory / "features.npy"
        label_file = layer_directory / "labels.npy"
        if not feature_file.is_file():
            raise FileNotFoundError(f"Missing feature file: {feature_file}")

        layer_features = np.load(feature_file)
        if layer_features.ndim != 2:
            raise ValueError(
                f"Expected 2-D features in {feature_file}, got {layer_features.shape}."
            )
        if expected_shape is None:
            expected_shape = layer_features.shape
        elif layer_features.shape != expected_shape:
            raise ValueError(
                f"Layer feature shape mismatch: expected {expected_shape}, "
                f"got {layer_features.shape} in {feature_file}."
            )

        if labels is None:
            if not label_file.is_file():
                raise FileNotFoundError(f"Missing label file: {label_file}")
            labels = np.load(label_file)

        all_features.append(layer_features)

    assert labels is not None
    stacked = np.stack(all_features, axis=1)
    if len(labels) != stacked.shape[0]:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match examples ({stacked.shape[0]})."
        )
    return stacked, labels


# Backward-compatible function name from the notebook.
load_all_layer_features = load_layerwise_features


def select_f1_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Select the probability threshold that maximizes F1 on validation data."""
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    precision, recall, thresholds = precision_recall_curve(
        labels_array,
        probabilities_array,
    )
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_index = int(np.nanargmax(f1_scores[:-1]))
    return float(thresholds[best_index])


def compute_probe_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    """Compute the metrics used in the original edge-probing notebook."""
    labels_array = np.asarray(labels).astype(int)
    probabilities_array = np.asarray(probabilities)
    predictions = (probabilities_array >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_array, probabilities_array)),
        "pr_auc": float(average_precision_score(labels_array, probabilities_array)),
    }


def train_scalar_mixing_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    probe_hidden: int = 256,
    dropout: float = 0.1,
    validation_size: float = 0.2,
    random_state: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
):
    """Train and validate the scalar-mixing specificity probe.

    The split reproduces the original notebook's 80/20 stratified split, but
    names the held-out portion ``validation`` to reflect its actual role in the
    experiment.  The validation set is used to check probe performance and to
    select the F1-maximizing threshold.  It is not the downstream SCOPEX data.
    """
    if features.ndim != 3:
        raise ValueError(
            f"features must have shape [N, layers, hidden], got {features.shape}."
        )
    if len(features) != len(labels):
        raise ValueError("features and labels must contain the same number of examples.")

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )

    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=resolved_device)
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=resolved_device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=resolved_device)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32, device=resolved_device)

    model = ScalarMixingEdgeProbe(
        num_layers=features.shape[1],
        hidden_dim=features.shape[2],
        probe_hidden=probe_hidden,
        dropout=dropout,
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(X_train_tensor), device=resolved_device)
        total_loss = 0.0

        for start in range(0, len(X_train_tensor), batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_features = X_train_tensor[batch_indices]
            batch_labels = y_train_tensor[batch_indices]

            logits, _ = model(batch_features)
            loss = loss_function(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_features)

        average_loss = total_loss / len(X_train_tensor)
        epoch_result = {"epoch": float(epoch), "train_loss": float(average_loss)}

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits, _ = model(X_val_tensor)
                probabilities = torch.sigmoid(logits).cpu().numpy()
            provisional_metrics = compute_probe_metrics(
                y_val,
                probabilities,
                threshold=0.5,
            )
            epoch_result.update(
                {
                    "val_accuracy_at_0.5": provisional_metrics["accuracy"],
                    "val_f1_at_0.5": provisional_metrics["f1"],
                    "val_roc_auc": provisional_metrics["roc_auc"],
                    "val_pr_auc": provisional_metrics["pr_auc"],
                }
            )
            if verbose:
                print(
                    f"Epoch {epoch:02d} | loss={average_loss:.4f} | "
                    f"val_acc={provisional_metrics['accuracy']:.3f} "
                    f"val_f1={provisional_metrics['f1']:.3f} "
                    f"val_auc={provisional_metrics['roc_auc']:.3f} "
                    f"val_pr={provisional_metrics['pr_auc']:.3f}"
                )
        history.append(epoch_result)

    model.eval()
    with torch.no_grad():
        validation_logits, mixing_weights = model(X_val_tensor)
        validation_probabilities = torch.sigmoid(validation_logits).cpu().numpy()

    threshold = select_f1_threshold(y_val, validation_probabilities)
    validation_metrics = compute_probe_metrics(
        y_val,
        validation_probabilities,
        threshold=threshold,
    )
    layer_weights = mixing_weights.detach().cpu().numpy().tolist()

    return model, validation_metrics, layer_weights, history


# Backward-compatible name from the original notebook.
def train_edge_probing(
    X_stacked,
    y,
    epochs=30,
    lr=1e-3,
    bs=64,
    probe_hidden=256,
    dropout=0.1,
    device=None,
):
    """Compatibility wrapper around :func:`train_scalar_mixing_probe`."""
    model, metrics, layer_weights, _ = train_scalar_mixing_probe(
        X_stacked,
        y,
        epochs=epochs,
        learning_rate=lr,
        batch_size=bs,
        probe_hidden=probe_hidden,
        dropout=dropout,
        device=device,
    )
    return model, metrics, layer_weights


def predict_specificity(
    model: ScalarMixingEdgeProbe,
    features: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a frozen probe and return probabilities and mixing weights.

    These continuous probabilities are the primary downstream specificity
    scores used when the trained probe is applied to SCOPEX.
    """
    resolved_device = torch.device(
        device if device is not None else next(model.parameters()).device
    )
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=resolved_device)

    model = model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        logits, mixing_weights = model(feature_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities, mixing_weights.cpu().numpy()


def save_probe_checkpoint(
    path: str | Path,
    model: ScalarMixingEdgeProbe,
    *,
    validation_metrics: dict[str, float],
    layer_weights: Sequence[float],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a probe checkpoint using the structure of the original notebook."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "metrics": dict(validation_metrics),
        "layer_weights": list(layer_weights),
        "config": {
            "num_layers": model.num_layers,
            "hidden_dim": model.hidden_dim,
            "probe_hidden": model.probe_hidden,
            "dropout": model.dropout,
        },
    }
    if extra:
        checkpoint["extra"] = extra

    torch.save(checkpoint, output_path)
    return output_path


def load_probe_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[ScalarMixingEdgeProbe, dict[str, Any]]:
    """Load a saved scalar-mixing probe and its checkpoint metadata."""
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(Path(path), map_location=resolved_device)
    config = checkpoint["config"]

    model = ScalarMixingEdgeProbe(
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        probe_hidden=config.get("probe_hidden", 256),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def summarize_layer_weights(
    weights: Sequence[float],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Return a machine-readable summary of learned scalar-mixing weights."""
    weights_array = np.asarray(weights, dtype=float)
    sorted_indices = np.argsort(weights_array)[::-1]
    return {
        "top_layers": [
            {
                "rank": rank,
                "layer": int(layer),
                "weight": float(weights_array[layer]),
                "percentage": float(weights_array[layer] * 100.0),
            }
            for rank, layer in enumerate(sorted_indices[:top_k], start=1)
        ],
        "mean": float(weights_array.mean()),
        "std": float(weights_array.std()),
        "max": float(weights_array.max()),
        "max_layer": int(weights_array.argmax()),
        "min": float(weights_array.min()),
        "min_layer": int(weights_array.argmin()),
    }
