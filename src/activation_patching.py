"""Reusable residual-stream activation patching utilities.

This module implements the model-agnostic core of Experiment 2B. Sentence
spans use Python's end-exclusive convention: ``[start, end)``.

The intervention is applied to the output of a decoder block, corresponding to
the post-MLP residual-stream state for Hugging Face decoder-only models such as
LLaMA, Mistral, and Qwen.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F


def get_transformer_layers(model) -> Sequence[torch.nn.Module]:
    """Return the decoder block sequence for a supported model architecture."""
    candidate_paths = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )

    for path in candidate_paths:
        current = model
        for attribute in path:
            if not hasattr(current, attribute):
                break
            current = getattr(current, attribute)
        else:
            return current

    raise ValueError(
        "Unsupported model architecture: decoder blocks were not found at "
        "model.model.layers, model.transformer.h, or model.gpt_neox.layers."
    )


def _split_block_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    """Separate a block's hidden-state tensor from any auxiliary outputs."""
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError("The first decoder-block output must be a tensor.")
        return output[0], output[1:]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected decoder-block output type: {type(output)!r}")
    return output, None


def _restore_block_output(
    hidden_states: torch.Tensor,
    auxiliary_outputs: tuple[Any, ...] | None,
) -> Any:
    if auxiliary_outputs is None:
        return hidden_states
    return (hidden_states,) + auxiliary_outputs


def _as_sequence_matrix(hidden_states: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return a ``(sequence, hidden)`` view for a batch-size-one tensor."""
    original_dimension = hidden_states.dim()
    if original_dimension == 3:
        if hidden_states.shape[0] != 1:
            raise ValueError("Activation patching currently requires batch size 1.")
        return hidden_states[0], original_dimension
    if original_dimension == 2:
        return hidden_states, original_dimension
    raise ValueError(
        f"Expected decoder-block states with 2 or 3 dimensions, got "
        f"shape {tuple(hidden_states.shape)}."
    )


def validate_span(start: int, end: int, sequence_length: int) -> tuple[int, int]:
    """Validate an end-exclusive token span."""
    start = int(start)
    end = int(end)
    if not 0 <= start < end <= sequence_length:
        raise ValueError(
            f"Invalid token span [{start}, {end}) for sequence length "
            f"{sequence_length}."
        )
    return start, end


def make_cache_hook(
    cache: MutableMapping[str, torch.Tensor],
    cache_key: str,
    span_start: int,
    span_end: int,
):
    """Cache decoder-block outputs over an end-exclusive token span."""

    def hook(_module, _inputs, output):
        hidden_states, _ = _split_block_output(output)
        sequence_states, _ = _as_sequence_matrix(hidden_states)
        start, end = validate_span(span_start, span_end, sequence_states.shape[0])
        cache[cache_key] = sequence_states[start:end].detach().clone()
        return output

    return hook


def make_patch_hook(
    cache: Mapping[str, torch.Tensor],
    cache_key: str,
    patch_start: int,
    patch_end: int,
):
    """Replace a target span with cached source activations.

    If source and target token spans differ slightly in length, only their
    common prefix is patched. The effective length should be recorded by the
    caller for transparency.
    """

    def hook(_module, _inputs, output):
        hidden_states, auxiliary_outputs = _split_block_output(output)
        sequence_states, original_dimension = _as_sequence_matrix(hidden_states)
        sequence_states = sequence_states.clone()

        start, end = validate_span(
            patch_start,
            patch_end,
            sequence_states.shape[0],
        )
        if cache_key not in cache:
            raise KeyError(f"Cached activation '{cache_key}' was not found.")

        cached_states = cache[cache_key].to(
            device=sequence_states.device,
            dtype=sequence_states.dtype,
        )
        if cached_states.dim() != 2:
            raise ValueError(
                f"Cached activation must be 2D, got {cached_states.dim()}D."
            )
        if cached_states.shape[1] != sequence_states.shape[1]:
            raise ValueError(
                "Source and target hidden dimensions do not match: "
                f"{cached_states.shape[1]} vs. {sequence_states.shape[1]}."
            )

        patch_length = min(end - start, cached_states.shape[0])
        if patch_length < 1:
            raise ValueError("The effective patch length is zero.")
        sequence_states[start : start + patch_length] = cached_states[:patch_length]

        if original_dimension == 3:
            new_hidden_states = hidden_states.clone()
            new_hidden_states[0] = sequence_states
        else:
            new_hidden_states = sequence_states

        return _restore_block_output(new_hidden_states, auxiliary_outputs)

    return hook


def make_span_mean_replacement_hook(span_start: int, span_end: int):
    """Replace each state in a span with that span's mean state."""

    def hook(_module, _inputs, output):
        hidden_states, auxiliary_outputs = _split_block_output(output)
        sequence_states, original_dimension = _as_sequence_matrix(hidden_states)
        sequence_states = sequence_states.clone()

        start, end = validate_span(span_start, span_end, sequence_states.shape[0])
        span = sequence_states[start:end]
        mean_state = span.mean(dim=0, keepdim=True)
        sequence_states[start:end] = mean_state.expand_as(span)

        if original_dimension == 3:
            new_hidden_states = hidden_states.clone()
            new_hidden_states[0] = sequence_states
        else:
            new_hidden_states = sequence_states

        return _restore_block_output(new_hidden_states, auxiliary_outputs)

    return hook


@torch.no_grad()
def token_log_probabilities(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return an aligned log-probability vector for one token sequence."""
    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(
        log_probabilities,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    aligned = torch.full(
        (input_ids.shape[1],),
        float("nan"),
        dtype=selected.dtype,
        device=selected.device,
    )
    aligned[1:] = selected[0]
    return aligned


def special_token_mask(tokenizer, token_ids: Sequence[int]) -> list[bool]:
    """Mark actual tokenizer special tokens without inventing a BOS token."""
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    return [int(token_id) in special_ids for token_id in token_ids]


@torch.no_grad()
def compute_span_surprisal(
    model,
    tokenizer,
    token_ids: Sequence[int],
    span_start: int,
    span_end: int,
    *,
    exclude_special_tokens: bool = True,
    log_base: float = 2.0,
) -> dict[str, float | int]:
    """Compute mean and summed in-situ surprisal for an end-exclusive span."""
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be positive and different from 1.")

    device = next(model.parameters()).device
    input_ids = torch.tensor([list(map(int, token_ids))], device=device)
    start, end = validate_span(span_start, span_end, input_ids.shape[1])
    log_probabilities = token_log_probabilities(model, input_ids)

    span_ids = list(map(int, token_ids[start:end]))
    mask = (
        special_token_mask(tokenizer, span_ids)
        if exclude_special_tokens
        else [False] * len(span_ids)
    )

    values: list[float] = []
    divisor = math.log(log_base)
    for log_probability, is_special in zip(
        log_probabilities[start:end].tolist(),
        mask,
    ):
        if is_special or math.isnan(float(log_probability)):
            continue
        values.append(-float(log_probability) / divisor)

    if not values:
        return {"mean": float("nan"), "sum": float("nan"), "n_tokens": 0}

    return {
        "mean": float(sum(values) / len(values)),
        "sum": float(sum(values)),
        "n_tokens": len(values),
    }


def run_single_patch(
    model,
    tokenizer,
    source_example: Mapping[str, Any],
    target_example: Mapping[str, Any],
    layer_index: int,
    *,
    source_start_key: str = "sentence_token_start",
    source_end_key: str = "sentence_token_end",
    target_start_key: str = "sentence_token_start",
    target_end_key: str = "sentence_token_end",
) -> dict[str, Any]:
    """Run one source-to-target residual-stream patch at one decoder layer."""
    source_ids = source_example["full_text_ids"]
    target_ids = target_example["full_text_ids"]
    source_start = int(source_example[source_start_key])
    source_end = int(source_example[source_end_key])
    target_start = int(target_example[target_start_key])
    target_end = int(target_example[target_end_key])

    source_start, source_end = validate_span(
        source_start,
        source_end,
        len(source_ids),
    )
    target_start, target_end = validate_span(
        target_start,
        target_end,
        len(target_ids),
    )

    source_length = source_end - source_start
    target_length = target_end - target_start
    effective_length = min(source_length, target_length)
    source_effective_end = source_start + effective_length
    target_effective_end = target_start + effective_length

    layers = get_transformer_layers(model)
    if not 0 <= layer_index < len(layers):
        raise IndexError(
            f"Layer {layer_index} is outside the valid range 0..{len(layers)-1}."
        )
    layer = layers[layer_index]

    cache: dict[str, torch.Tensor] = {}
    device = next(model.parameters()).device
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)

    cache_handle = layer.register_forward_hook(
        make_cache_hook(
            cache,
            "source",
            source_start,
            source_effective_end,
        )
    )
    try:
        with torch.no_grad():
            model(input_ids=source_tensor, use_cache=False)
    finally:
        cache_handle.remove()

    baseline = compute_span_surprisal(
        model,
        tokenizer,
        target_ids,
        target_start,
        target_effective_end,
    )

    patch_handle = layer.register_forward_hook(
        make_patch_hook(
            cache,
            "source",
            target_start,
            target_effective_end,
        )
    )
    try:
        patched = compute_span_surprisal(
            model,
            tokenizer,
            target_ids,
            target_start,
            target_effective_end,
        )
    finally:
        patch_handle.remove()

    return {
        "layer_idx": int(layer_index),
        "source_id": source_example.get("id", source_example.get("idx")),
        "target_id": target_example.get("id", target_example.get("idx")),
        "source_case": source_example.get("case"),
        "target_case": target_example.get("case"),
        "baseline_mean": baseline["mean"],
        "patched_mean": patched["mean"],
        "delta_mean": float(patched["mean"] - baseline["mean"]),
        "n_tokens": baseline["n_tokens"],
        "source_span_length": source_length,
        "target_span_length": target_length,
        "effective_patch_length": effective_length,
    }


def run_single_span_mean_replacement(
    model,
    tokenizer,
    target_example: Mapping[str, Any],
    layer_index: int,
    *,
    span_start_key: str = "sentence_token_start",
    span_end_key: str = "sentence_token_end",
) -> dict[str, Any]:
    """Run the span-mean replacement baseline for one item and layer."""
    token_ids = target_example["full_text_ids"]
    start = int(target_example[span_start_key])
    end = int(target_example[span_end_key])
    start, end = validate_span(start, end, len(token_ids))

    layers = get_transformer_layers(model)
    if not 0 <= layer_index < len(layers):
        raise IndexError(
            f"Layer {layer_index} is outside the valid range 0..{len(layers)-1}."
        )

    baseline = compute_span_surprisal(
        model,
        tokenizer,
        token_ids,
        start,
        end,
    )

    handle = layers[layer_index].register_forward_hook(
        make_span_mean_replacement_hook(start, end)
    )
    try:
        replaced = compute_span_surprisal(
            model,
            tokenizer,
            token_ids,
            start,
            end,
        )
    finally:
        handle.remove()

    return {
        "layer_idx": int(layer_index),
        "target_id": target_example.get("id", target_example.get("idx")),
        "target_case": target_example.get("case"),
        "baseline_mean": baseline["mean"],
        "replaced_mean": replaced["mean"],
        "delta_mean": float(replaced["mean"] - baseline["mean"]),
        "n_tokens": baseline["n_tokens"],
        "span_length": end - start,
    }
