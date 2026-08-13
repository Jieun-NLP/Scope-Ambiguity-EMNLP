#!/usr/bin/env python3
"""Extract target-NP representations from the SPECIFICITY dataset for Exp. 2C.

This script prepares the layer-wise input features used to train the
scalar-mixing specificity edge probe in Experiment 2C.

Unlike Experiments 2A and 2B, which operate on the full target sentence,
Experiment 2C extracts representations only from the annotated target NP.
The SPECIFICITY dataset already provides character-level NP boundaries
(`span_char_start`, `span_char_end`), so those annotations are treated as the
source of truth. They are converted to end-exclusive token spans using the
tokenizer's offset mapping, and the hidden states over the NP span are
mean-pooled independently at each layer.

Output layout (compatible with the original probe-training notebook):

    OUTPUT_DIR/
        layer_00/
            features.npy
            labels.npy
            metadata.jsonl
        layer_01/
            ...
        ...
        extraction_summary.json

Labels are encoded as:
    SPECIFIC     -> 1
    NON-SPECIFIC -> 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_representation import (  # noqa: E402
    ensure_directory,
    extract_mean_pooled_span_representation,
    load_json_records,
    load_model_and_tokenizer,
    select_hidden_state_layers,
)
from src.target_span import (  # noqa: E402
    char_span_to_token_span,
    validate_token_span,
)


LABEL_TO_ID = {
    "NON-SPECIFIC": 0,
    "SPECIFIC": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract layer-wise, mean-pooled target-NP representations from "
            "the SPECIFICITY dataset for Experiment 2C."
        )
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help=(
            "SPECIFICITY dataset in JSONL or JSON format. Each record must "
            "contain full_text, target_np, label, span_char_start, and "
            "span_char_end."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory in which layer_XX/features.npy, labels.npy, and "
            "metadata.jsonl will be written."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--model-type",
        choices=("causal_lm", "base"),
        default="causal_lm",
        help=(
            "Use 'causal_lm' for LLaMA, Mistral, and Qwen, or 'base' for "
            "encoder models such as BERT."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Hidden-state indices to extract. Omit to extract embedding layer "
            "0 and all transformer block outputs."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Optional dtype used when loading the language model.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Numeric dtype used for saved feature arrays.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help=(
            "Maximum tokenized sequence length. This preserves the 4096-token "
            "limit used in the original extraction notebook."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model code when required.",
    )
    parser.add_argument(
        "--allow-invalid-spans",
        action="store_true",
        help=(
            "Skip records with invalid or truncated target spans instead of "
            "raising an error. Skipped records are reported in the summary."
        ),
    )
    return parser.parse_args()


def convert_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None

    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def encode_label(value: Any) -> int:
    """Convert the dataset's string specificity label to 0/1."""
    normalized = str(value).strip().upper()
    if normalized not in LABEL_TO_ID:
        raise ValueError(
            f"Unknown specificity label {value!r}; expected one of "
            f"{sorted(LABEL_TO_ID)}."
        )
    return LABEL_TO_ID[normalized]


def validate_record(record: dict[str, Any]) -> None:
    """Check fields required for target-NP feature extraction."""
    required = (
        "full_text",
        "target_np",
        "label",
        "span_char_start",
        "span_char_end",
    )
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")

    full_text = record["full_text"]
    target_np = record["target_np"]
    if not isinstance(full_text, str) or not full_text:
        raise ValueError("'full_text' must be a non-empty string.")
    if not isinstance(target_np, str) or not target_np:
        raise ValueError("'target_np' must be a non-empty string.")

    char_start = int(record["span_char_start"])
    char_end = int(record["span_char_end"])
    if not (0 <= char_start < char_end <= len(full_text)):
        raise ValueError(
            f"Invalid character span [{char_start}, {char_end}) for text "
            f"length {len(full_text)}."
        )

    annotated_text = full_text[char_start:char_end]
    if annotated_text != target_np:
        raise ValueError(
            "The annotated character span does not exactly match target_np: "
            f"span={annotated_text!r}, target_np={target_np!r}."
        )


def get_record_identifier(record: dict[str, Any], fallback_index: int) -> str:
    """Return a stable item identifier for metadata and diagnostics."""
    identifier = record.get("id", record.get("idx"))
    if identifier is None:
        return str(fallback_index)
    return str(identifier)


def build_metadata(
    record: dict[str, Any],
    *,
    item_id: str,
    label_id: int,
    char_span: tuple[int, int],
    token_span: tuple[int, int],
    decoded_text: str,
) -> dict[str, Any]:
    """Create public metadata aligned with the original extraction notebook."""
    return {
        "id": item_id,
        "label": label_id,
        "label_text": str(record["label"]),
        "case": record.get("case"),
        "context": record.get("context", ""),
        "sentence": record.get("sentence", ""),
        "full_text": record["full_text"],
        "target_np": record["target_np"],
        "span_char_start": int(char_span[0]),
        "span_char_end": int(char_span[1]),
        "span_token_start": int(token_span[0]),
        "span_token_end": int(token_span[1]),
        "decoded_target_span": decoded_text,
    }


def extract_specificity_representations(
    records: list[dict[str, Any]],
    model,
    tokenizer,
    *,
    layers: list[int],
    device,
    output_dtype: np.dtype,
    max_length: int,
    allow_invalid_spans: bool,
) -> tuple[
    dict[int, list[np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Extract mean-pooled target-NP vectors for all requested hidden states."""
    layer_features: dict[int, list[np.ndarray]] = {
        layer: [] for layer in layers
    }
    metadata: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record_index, record in enumerate(
        tqdm(records, desc="Extracting target-NP representations"),
        start=1,
    ):
        item_id = get_record_identifier(record, record_index)

        try:
            validate_record(record)

            full_text = record["full_text"]
            target_np = record["target_np"]
            char_span = (
                int(record["span_char_start"]),
                int(record["span_char_end"]),
            )
            label_id = encode_label(record["label"])

            # Tokenize with offsets for both character-to-token mapping and
            # the model forward pass. The character-span annotations in the
            # SPECIFICITY dataset are the source of truth.
            model_encoding = tokenizer(
                full_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            model_offsets = model_encoding.pop("offset_mapping")[0].tolist()

            # Use offsets from the actual (possibly truncated) model input so
            # a target NP outside max_length is detected rather than silently
            # mapped to a different position.
            _token_indices, _snapped_char_span, token_span = (
                char_span_to_token_span(
                    full_text,
                    char_span,
                    model_offsets,
                    mode="snap_out",
                    trim_whitespace=True,
                )
            )
            if token_span is None:
                raise ValueError(
                    "The annotated target NP does not overlap any token in the "
                    "model input. It may have been truncated by --max-length."
                )

            # Validate that the token span decodes back to the annotated NP.
            validation_encoding = {
                "input_ids": model_encoding["input_ids"][0].tolist()
            }
            validation = validate_token_span(
                encoding=validation_encoding,
                tokenizer=tokenizer,
                target_np=target_np,
                token_span=token_span,
                case_sensitive=False,
                normalize_whitespace=True,
                normalize_quotes=True,
            )
            if not validation["decoded_well"]:
                raise ValueError(
                    "Token-span validation failed: "
                    f"target_np={target_np!r}, "
                    f"decoded={validation['decoded_text']!r}, "
                    f"char_span={char_span}, token_span={token_span}."
                )

            input_ids = model_encoding["input_ids"]
            attention_mask = model_encoding.get("attention_mask")

            layer_vectors = extract_mean_pooled_span_representation(
                model,
                input_ids,
                span_start=token_span[0],
                span_end=token_span[1],
                layers=layers,
                attention_mask=attention_mask,
                device=device,
                output_dtype=output_dtype,
            )

            for layer in layers:
                vector = np.asarray(layer_vectors[layer])
                if vector.ndim != 1:
                    vector = vector.reshape(-1)
                layer_features[layer].append(vector)

            metadata.append(
                build_metadata(
                    record,
                    item_id=item_id,
                    label_id=label_id,
                    char_span=char_span,
                    token_span=token_span,
                    decoded_text=validation["decoded_text"],
                )
            )

        except Exception as error:
            if not allow_invalid_spans:
                raise RuntimeError(
                    f"Failed to process record {record_index} "
                    f"(id={item_id!r})."
                ) from error

            skipped.append(
                {
                    "record_index": record_index,
                    "id": item_id,
                    "error": str(error),
                }
            )

    return layer_features, metadata, skipped


def save_layerwise_dataset(
    *,
    layer_features: dict[int, list[np.ndarray]],
    metadata: list[dict[str, Any]],
    layers: list[int],
    output_dir: Path,
    model_name: str,
    output_dtype_name: str,
    skipped: list[dict[str, Any]],
) -> Path:
    """Save features, labels, metadata, and an extraction summary."""
    destination = ensure_directory(output_dir)

    if not metadata:
        raise ValueError("No valid examples were extracted.")

    labels = np.asarray(
        [int(item["label"]) for item in metadata],
        dtype=np.int32,
    )

    feature_shapes: dict[str, list[int]] = {}

    for layer in layers:
        vectors = layer_features[layer]
        if len(vectors) != len(metadata):
            raise RuntimeError(
                f"Layer {layer} contains {len(vectors)} vectors but "
                f"{len(metadata)} metadata records were retained."
            )

        layer_dir = ensure_directory(destination / f"layer_{layer:02d}")
        features = np.stack(vectors, axis=0)

        np.save(layer_dir / "features.npy", features)
        np.save(layer_dir / "labels.npy", labels)

        with (layer_dir / "metadata.jsonl").open(
            "w", encoding="utf-8"
        ) as output_file:
            for item in metadata:
                output_file.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )

        feature_shapes[str(layer)] = list(features.shape)

    summary = {
        "model": model_name,
        "pooling": "span-mean",
        "target_unit": "annotated target NP",
        "label_mapping": LABEL_TO_ID,
        "num_examples": len(metadata),
        "num_skipped": len(skipped),
        "layers": layers,
        "output_dtype": output_dtype_name,
        "feature_shapes": feature_shapes,
        "skipped_records": skipped,
    }

    summary_path = destination / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    return summary_path


def main() -> None:
    args = parse_args()
    records = load_json_records(args.input_data)

    model, tokenizer, device = load_model_and_tokenizer(
        args.model,
        model_type=args.model_type,
        device=args.device,
        torch_dtype=convert_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=True,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Experiment 2C target-span extraction requires a fast tokenizer "
            "with offset mappings."
        )

    layers = select_hidden_state_layers(
        model.config.num_hidden_layers,
        args.layers,
    )
    output_dtype = {
        "float32": np.float32,
        "float16": np.float16,
    }[args.output_dtype]

    layer_features, metadata, skipped = extract_specificity_representations(
        records,
        model,
        tokenizer,
        layers=layers,
        device=device,
        output_dtype=output_dtype,
        max_length=args.max_length,
        allow_invalid_spans=args.allow_invalid_spans,
    )

    summary_path = save_layerwise_dataset(
        layer_features=layer_features,
        metadata=metadata,
        layers=layers,
        output_dir=args.output_dir,
        model_name=args.model,
        output_dtype_name=args.output_dtype,
        skipped=skipped,
    )

    positives = sum(int(item["label"]) for item in metadata)
    negatives = len(metadata) - positives

    print(f"Input records: {len(records)}")
    print(f"Extracted examples: {len(metadata)}")
    print(f"Skipped examples: {len(skipped)}")
    print(f"Labels: SPECIFIC={positives}, NON-SPECIFIC={negatives}")
    print(f"Hidden-state indices: {layers}")
    print("Pooling: span-mean over the annotated target NP")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary: {summary_path}")
    print(
        "Next step: run scripts/train_specificity_probe_for_Exp2C.py "
        "on this output directory."
    )


if __name__ == "__main__":
    main()
