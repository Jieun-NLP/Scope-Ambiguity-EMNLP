#!/usr/bin/env python3
"""Preprocessing for Experiment 2A: extract target-sentence representations.

This script MUST be run before ``Experiment2.py``. It reads tokenized records
containing target-sentence token boundaries, extracts hidden states over the
target sentence at every requested layer, mean-pools the span, and writes
``sentence_means.npz``.

Pipeline:
    extract_sentence_representation_for_Exp2A.py
        -> sentence_means.npz
        -> Experiment2.py
        -> layer-wise cosine similarity and CKA
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_representation import (  # noqa: E402
    extract_mean_pooled_span_representation,
    get_span_from_record,
    load_json_records,
    load_model_and_tokenizer,
    save_layerwise_representations,
    select_hidden_state_layers,
)


SENTENCE_START_KEYS = (
    "sentence_token_start_with_special",
    "sentence_token_start",
)
SENTENCE_END_KEYS = (
    "sentence_token_end_with_special",
    "sentence_token_end",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract mean-pooled target-sentence hidden representations "
            "required by Experiment 2A."
        )
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Processed JSON or JSONL file containing token IDs and sentence spans.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Destination NPZ file, normally sentence_means.npz.",
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
            "Use 'causal_lm' for LLaMA, Mistral, and Qwen, "
            "or 'base' for encoder models such as BERT."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Hidden-state indices to extract. Omit to extract embedding layer 0 "
            "and all transformer block outputs."
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
        help="Optional model-loading dtype.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Numeric dtype used in the saved NPZ archive.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model code when required.",
    )
    return parser.parse_args()


def normalize_case_label(case: Any) -> str:
    """Convert numeric and string condition labels to stable strings."""
    if isinstance(case, bool):
        return str(case)
    if isinstance(case, (int, np.integer)):
        return str(int(case))
    if isinstance(case, (float, np.floating)) and float(case).is_integer():
        return str(int(case))
    return str(case)


def get_record_identifier(record: dict[str, Any]) -> str:
    """Read the item identifier used in the output NPZ key."""
    identifier = record.get("idx", record.get("id"))
    if identifier is None:
        raise KeyError("Each record must contain either 'idx' or 'id'.")
    return str(identifier)


def get_record_case(record: dict[str, Any]) -> str:
    """Read and normalize the experimental condition label."""
    if "case" not in record:
        raise KeyError("Each record must contain a 'case' field.")
    return normalize_case_label(record["case"])


def get_input_ids(record: dict[str, Any]) -> list[int]:
    """Read pretokenized input IDs from a processed record."""
    token_ids = record.get("full_text_ids")
    if not token_ids:
        raise KeyError(
            "Each record must contain a non-empty 'full_text_ids' field."
        )
    return [int(token_id) for token_id in token_ids]


def convert_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None

    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def extract_sentence_representations(
    records: list[dict[str, Any]],
    model,
    *,
    layers: list[int],
    device,
    output_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Extract mean-pooled target-sentence representations for Experiment 2A."""
    representations: dict[str, np.ndarray] = {}

    for record_index, record in enumerate(
        tqdm(records, desc="Extracting target-sentence representations"),
        start=1,
    ):
        try:
            item_id = get_record_identifier(record)
            case = get_record_case(record)
            input_ids = get_input_ids(record)
            span_start, span_end = get_span_from_record(
                record,
                start_keys=SENTENCE_START_KEYS,
                end_keys=SENTENCE_END_KEYS,
                sequence_length=len(input_ids),
            )

            layer_vectors = extract_mean_pooled_span_representation(
                model,
                input_ids,
                span_start=span_start,
                span_end=span_end,
                layers=layers,
                device=device,
                output_dtype=output_dtype,
            )

            for layer, vector in layer_vectors.items():
                key = f"{item_id}_{case}_L{layer}"
                if key in representations:
                    raise ValueError(f"Duplicate representation key: {key}")
                representations[key] = vector

        except Exception as error:
            raise RuntimeError(
                f"Failed to process record {record_index}."
            ) from error

    return representations


def main() -> None:
    args = parse_args()
    records = load_json_records(args.input_data)

    model, _tokenizer, device = load_model_and_tokenizer(
        args.model,
        model_type=args.model_type,
        device=args.device,
        torch_dtype=convert_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )

    layers = select_hidden_state_layers(
        model.config.num_hidden_layers,
        args.layers,
    )
    output_dtype = {
        "float32": np.float32,
        "float16": np.float16,
    }[args.output_dtype]

    representations = extract_sentence_representations(
        records,
        model,
        layers=layers,
        device=device,
        output_dtype=output_dtype,
    )
    output_path = save_layerwise_representations(
        representations,
        args.output_file,
    )

    print(f"Processed records: {len(records)}")
    print(f"Saved representations: {len(representations)}")
    print(f"Extracted hidden-state indices: {layers}")
    print(f"Output file: {output_path}")
    print("Next step: run scripts/Experiment2.py using this NPZ file.")


if __name__ == "__main__":
    main()
