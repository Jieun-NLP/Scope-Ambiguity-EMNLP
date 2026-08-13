#!/usr/bin/env python3
"""Train the scalar-mixing specificity edge probe for Experiment 2C.

This script trains the main edge probe reported in Experiment 2C. It consumes
the layer-wise target-NP representations produced by

    scripts/extract_specificity_representation_for_Exp2C.py

and stacks them into an array of shape [N, L, H], where:

    N = number of SPECIFICITY examples
    L = number of hidden-state layers
    H = language-model hidden dimension

The probe follows the original experiment:

1. Learn a scalar weight for each hidden-state layer.
2. Normalize the weights with softmax.
3. Form a weighted sum of the layer-wise target-NP representations.
4. Feed the mixed representation to a two-layer MLP.
5. Jointly optimize the mixing weights and MLP with binary cross-entropy.
6. Use an 80/20 stratified training/validation split.
7. Select the decision threshold on the validation split by maximizing F1.
8. Save the trained probe for later frozen inference on SCOPEX.

The held-out 20% is a validation split, not the downstream SCOPEX evaluation
data. SCOPEX is never used to optimize the specificity probe.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.probing import (  # noqa: E402
    load_layerwise_features,
    save_probe_checkpoint,
    summarize_layer_weights,
    train_scalar_mixing_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and validate the scalar-mixing specificity edge probe "
            "used in Experiment 2C."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help=(
            "Directory produced by "
            "extract_specificity_representation_for_Exp2C.py. "
            "It must contain layer_XX/features.npy and labels.npy files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which the trained probe and training summaries are saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs. Original experiment: 30.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate. Original experiment: 1e-3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size. Original experiment: 64.",
    )
    parser.add_argument(
        "--probe-hidden",
        type=int,
        default=256,
        help="Hidden width of the two-layer MLP probe. Original experiment: 256.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability in the MLP probe. Original experiment: 0.1.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help=(
            "Fraction of the independent SPECIFICITY dataset held out for "
            "validation and threshold selection. Original experiment: 0.2."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the stratified split and PyTorch training.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="specificity_edge_probe.pt",
        help="Filename for the saved probe checkpoint.",
    )
    parser.add_argument(
        "--top-k-layers",
        type=int,
        default=10,
        help="Number of highest-weighted layers to include in the summary.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    """Resolve the training device."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    validation_size: float,
) -> None:
    """Validate feature shape, labels, and split configuration."""
    if features.ndim != 3:
        raise ValueError(
            "Expected features with shape [N, layers, hidden_dim], "
            f"got {features.shape}."
        )

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if len(features) != len(labels):
        raise ValueError(
            f"Feature/label size mismatch: {len(features)} examples vs. "
            f"{len(labels)} labels."
        )

    unique_labels = sorted(np.unique(labels).tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            "Experiment 2C expects binary labels encoded as "
            f"NON-SPECIFIC=0 and SPECIFIC=1; found {unique_labels}."
        )

    if not 0.0 < validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1.")


def write_json(path: Path, data) -> None:
    """Write a JSON object using a public-repository-friendly format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading layer-wise SPECIFICITY features...")
    features, labels = load_layerwise_features(args.feature_dir)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    validate_inputs(features, labels, args.validation_size)

    num_examples, num_layers, hidden_dim = features.shape
    num_specific = int(labels.sum())
    num_non_specific = int(len(labels) - num_specific)

    print(
        f"Feature tensor: N={num_examples}, layers={num_layers}, "
        f"hidden_dim={hidden_dim}"
    )
    print(
        f"Labels: SPECIFIC={num_specific}, "
        f"NON-SPECIFIC={num_non_specific}"
    )
    print(
        f"Split: {(1.0 - args.validation_size):.0%} training / "
        f"{args.validation_size:.0%} validation"
    )
    print(f"Device: {device}")
    print("Training scalar-mixing specificity probe...")

    model, validation_metrics, layer_weights, history = (
        train_scalar_mixing_probe(
            features,
            labels,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            probe_hidden=args.probe_hidden,
            dropout=args.dropout,
            validation_size=args.validation_size,
            random_state=args.seed,
            device=device,
            verbose=True,
        )
    )

    checkpoint_path = args.output_dir / args.checkpoint_name
    save_probe_checkpoint(
        checkpoint_path,
        model,
        validation_metrics=validation_metrics,
        layer_weights=layer_weights,
        extra={
            "feature_dir": str(args.feature_dir),
            "num_examples": num_examples,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "label_mapping": {
                "NON-SPECIFIC": 0,
                "SPECIFIC": 1,
            },
            "training": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "probe_hidden": args.probe_hidden,
                "dropout": args.dropout,
                "validation_size": args.validation_size,
                "seed": args.seed,
            },
        },
    )

    layer_summary = summarize_layer_weights(
        layer_weights,
        top_k=min(args.top_k_layers, len(layer_weights)),
    )

    results = {
        "probe": "scalar-mixing two-layer MLP",
        "dataset_role": (
            "Independent SPECIFICITY training/validation dataset; "
            "SCOPEX is not used for probe optimization."
        ),
        "feature_shape": {
            "num_examples": num_examples,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        },
        "label_counts": {
            "SPECIFIC": num_specific,
            "NON-SPECIFIC": num_non_specific,
        },
        "validation_metrics": validation_metrics,
        "layer_weight_summary": layer_summary,
        "layer_weights": [float(weight) for weight in layer_weights],
        "checkpoint": str(checkpoint_path),
    }

    write_json(args.output_dir / "validation_results.json", results)
    write_json(args.output_dir / "training_history.json", history)

    print("\nValidation results")
    print("------------------")
    print(f"Threshold : {validation_metrics['threshold']:.6f}")
    print(f"Accuracy  : {validation_metrics['accuracy']:.4f}")
    print(f"F1        : {validation_metrics['f1']:.4f}")
    print(f"Precision : {validation_metrics['precision']:.4f}")
    print(f"Recall    : {validation_metrics['recall']:.4f}")
    print(f"ROC-AUC   : {validation_metrics['roc_auc']:.4f}")
    print(f"PR-AUC    : {validation_metrics['pr_auc']:.4f}")

    print("\nHighest scalar-mixing weights")
    print("-----------------------------")
    for item in layer_summary["top_layers"]:
        print(
            f"{item['rank']:>2}. Layer {item['layer']:>2}: "
            f"{item['percentage']:.3f}%"
        )

    print(f"\nCheckpoint: {checkpoint_path}")
    print(
        f"Validation summary: "
        f"{args.output_dir / 'validation_results.json'}"
    )
    print(
        f"Training history: "
        f"{args.output_dir / 'training_history.json'}"
    )
    print(
        "\nThe trained probe is now ready for frozen inference on SCOPEX. "
        "Do not re-optimize the probe or its threshold on SCOPEX."
    )


if __name__ == "__main__":
    main()
