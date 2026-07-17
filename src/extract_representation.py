"""Reusable utilities for extracting hidden representations from token spans.

The functions in this module are span-agnostic. A span may correspond to a
sentence, noun phrase, verb phrase, pronoun, or any other token interval,
provided that its start and end indices are supplied in token coordinates.

Layer indexing follows Hugging Face's ``output_hidden_states=True`` convention:
layer 0 is the embedding output, and layers 1..N are transformer block outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from either JSONL or a JSON array."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig") as input_file:
        first_non_whitespace = ""
        while True:
            character = input_file.read(1)
            if not character:
                break
            if not character.isspace():
                first_non_whitespace = character
                break
        input_file.seek(0)

        if first_non_whitespace == "[":
            records = json.load(input_file)
            if not isinstance(records, list):
                raise ValueError("A JSON input file must contain a top-level list.")
            return records

        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}, "
                    f"but found {type(record).__name__}."
                )
            records.append(record)

    return records


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve an explicit device or select CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_huggingface_token(environment_variable: str = "HF_TOKEN") -> str | None:
    """Read a Hugging Face token from an environment variable."""
    return os.getenv(environment_variable)


def load_model_and_tokenizer(
    model_name: str,
    *,
    model_type: str = "causal_lm",
    device: str | None = None,
    torch_dtype: torch.dtype | str | None = None,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
    token_environment_variable: str = "HF_TOKEN",
):
    """Load a Hugging Face model and tokenizer without hard-coded credentials.

    Parameters
    ----------
    model_type:
        ``"causal_lm"`` for autoregressive language models such as LLaMA,
        Mistral, and Qwen, or ``"base"`` for encoder models such as BERT.
    """
    resolved_device = resolve_device(device)
    token = get_huggingface_token(token_environment_variable)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
    )

    model_class = {
        "causal_lm": AutoModelForCausalLM,
        "base": AutoModel,
    }.get(model_type)
    if model_class is None:
        raise ValueError("model_type must be either 'causal_lm' or 'base'.")

    model_kwargs: dict[str, Any] = {
        "token": token,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = model_class.from_pretrained(model_name, **model_kwargs)
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def select_hidden_state_layers(
    number_of_transformer_layers: int,
    requested_layers: Sequence[int] | None = None,
) -> list[int]:
    """Return validated hidden-state indices.

    The returned range includes the embedding output at index 0 and the final
    transformer block at index ``number_of_transformer_layers``.
    """
    maximum_index = int(number_of_transformer_layers)

    if requested_layers is None:
        return list(range(maximum_index + 1))

    layers = sorted(set(int(layer) for layer in requested_layers))
    invalid = [layer for layer in layers if layer < 0 or layer > maximum_index]
    if invalid:
        raise ValueError(
            f"Invalid hidden-state layer indices {invalid}; "
            f"valid indices are 0 through {maximum_index}."
        )
    return layers


def normalize_token_span(
    start: int | None,
    end: int | None,
    sequence_length: int,
    *,
    fallback_to_full_sequence: bool = False,
) -> tuple[int, int]:
    """Validate and normalize an end-exclusive token span."""
    normalized_start = 0 if start is None else int(start)
    normalized_end = sequence_length if end is None else int(end)

    normalized_start = max(0, min(normalized_start, sequence_length))
    normalized_end = max(0, min(normalized_end, sequence_length))

    if normalized_end <= normalized_start:
        if fallback_to_full_sequence:
            return 0, sequence_length
        raise ValueError(
            f"Invalid token span [{normalized_start}, {normalized_end}) "
            f"for sequence length {sequence_length}."
        )

    return normalized_start, normalized_end


def get_span_from_record(
    record: Mapping[str, Any],
    *,
    start_keys: Sequence[str],
    end_keys: Sequence[str],
    sequence_length: int,
    fallback_to_full_sequence: bool = False,
) -> tuple[int, int]:
    """Read a token span from the first available start and end keys."""

    def first_available(keys: Sequence[str]) -> Any:
        for key in keys:
            if key in record and record[key] is not None:
                return record[key]
        return None

    start = first_available(start_keys)
    end = first_available(end_keys)

    if start is None or end is None:
        missing = []
        if start is None:
            missing.append(f"start keys {list(start_keys)}")
        if end is None:
            missing.append(f"end keys {list(end_keys)}")
        raise KeyError(
            "The record does not contain the required token span: "
            + " and ".join(missing)
        )

    return normalize_token_span(
        start,
        end,
        sequence_length,
        fallback_to_full_sequence=fallback_to_full_sequence,
    )


@torch.no_grad()
def extract_span_hidden_states(
    model,
    input_ids: torch.Tensor | Sequence[int],
    *,
    span_start: int,
    span_end: int,
    layers: Sequence[int] | None = None,
    attention_mask: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    output_dtype: np.dtype = np.float32,
) -> dict[int, np.ndarray]:
    """Extract token-level hidden states for an arbitrary end-exclusive span.

    Returns
    -------
    dict
        Mapping from hidden-state layer index to an array of shape
        ``(span_length, hidden_size)``.
    """
    resolved_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([list(map(int, input_ids))], dtype=torch.long)
    elif input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must represent exactly one sequence.")

    input_ids = input_ids.to(resolved_device)
    if attention_mask is not None:
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(resolved_device)

    sequence_length = int(input_ids.shape[1])
    normalized_start, normalized_end = normalize_token_span(
        span_start,
        span_end,
        sequence_length,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("The model did not return hidden states.")

    selected_layers = (
        list(range(len(hidden_states)))
        if layers is None
        else [int(layer) for layer in layers]
    )

    invalid_layers = [
        layer for layer in selected_layers
        if layer < 0 or layer >= len(hidden_states)
    ]
    if invalid_layers:
        raise ValueError(
            f"Requested hidden-state indices {invalid_layers}, "
            f"but the model returned indices 0 through {len(hidden_states) - 1}."
        )

    extracted: dict[int, np.ndarray] = {}
    for layer in selected_layers:
        span_tensor = hidden_states[layer][0, normalized_start:normalized_end, :]
        extracted[layer] = (
            span_tensor.detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
            .astype(output_dtype, copy=False)
        )

    return extracted


def mean_pool_span_hidden_states(
    span_hidden_states: Mapping[int, np.ndarray],
    *,
    output_dtype: np.dtype = np.float32,
) -> dict[int, np.ndarray]:
    """Mean-pool token-level span representations independently by layer."""
    pooled: dict[int, np.ndarray] = {}

    for layer, vectors in span_hidden_states.items():
        array = np.asarray(vectors)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(
                f"Layer {layer} must contain a non-empty "
                "(span_length, hidden_size) array."
            )
        pooled[int(layer)] = array.mean(axis=0).astype(output_dtype, copy=False)

    return pooled


@torch.no_grad()
def extract_mean_pooled_span_representation(
    model,
    input_ids: torch.Tensor | Sequence[int],
    *,
    span_start: int,
    span_end: int,
    layers: Sequence[int] | None = None,
    attention_mask: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    output_dtype: np.dtype = np.float32,
) -> dict[int, np.ndarray]:
    """Extract and mean-pool an arbitrary token span for selected layers."""
    span_states = extract_span_hidden_states(
        model,
        input_ids,
        span_start=span_start,
        span_end=span_end,
        layers=layers,
        attention_mask=attention_mask,
        device=device,
        output_dtype=output_dtype,
    )
    return mean_pool_span_hidden_states(
        span_states,
        output_dtype=output_dtype,
    )


def save_layerwise_representations(
    representations: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Save keyed representation vectors as a compressed NPZ archive."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not representations:
        raise ValueError("No representations were supplied for saving.")

    np.savez_compressed(
        destination,
        **{
            str(key): np.asarray(vector)
            for key, vector in representations.items()
        },
    )
    return destination
