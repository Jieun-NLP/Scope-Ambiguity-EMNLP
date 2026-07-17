#!/usr/bin/env python3
"""Experiment 2B: target-sentence residual-stream activation patching.

The same protocol is used for LLaMA, Mistral, and Qwen. Model-specific
architectural handling is confined to ``src.activation_patching``.

For each aligned condition pair, this script patches the post-MLP residual
states of the complete target-sentence span in both directions and measures the
change in target-sentence surprisal. It also runs the span-mean replacement
baseline reported in the paper.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.activation_patching import (  # noqa: E402
    get_transformer_layers,
    run_single_patch,
    run_single_span_mean_replacement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2B: run bidirectional target-sentence residual-stream "
            "activation patching and a span-mean replacement baseline."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID.")
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Processed JSON or JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV files, repaired data, and figures.",
    )
    parser.add_argument(
        "--source-case",
        default="0",
        help="First condition label. Default: 0.",
    )
    parser.add_argument(
        "--target-case",
        default="1",
        help="Second condition label. Default: 1.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Decoder layers to test. Omit to use every decoder layer.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Optional number of aligned item pairs for a test run.",
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
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model code when required.",
    )
    parser.add_argument(
        "--skip-span-repair",
        action="store_true",
        help="Use stored token IDs and sentence spans without retokenizing.",
    )
    parser.add_argument(
        "--skip-span-mean-baseline",
        action="store_true",
        help="Skip the span-mean replacement baseline.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Save numeric results without generating a figure.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input data file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as input_file:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in input_file if line.strip()]
        if path.suffix.lower() == ".json":
            records = json.load(input_file)
            if not isinstance(records, list):
                raise ValueError("A JSON dataset must contain a top-level list.")
            return records
    raise ValueError("Input data must use the .json or .jsonl extension.")


def normalize_case_label(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def item_identifier(record: dict[str, Any]) -> str:
    identifier = record.get("id", record.get("idx"))
    if identifier is None:
        raise KeyError("Each record must contain either 'id' or 'idx'.")
    return str(identifier)


def normalize_text_for_alignment(text: str) -> str:
    """Normalize harmless whitespace and quotation-mark variation."""
    normalized = text.strip()
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def find_sentence_character_span(full_text: str, sentence: str) -> tuple[int, int]:
    """Find the sentence in the full sequence using conservative fallbacks."""
    start = full_text.find(sentence)
    if start >= 0:
        return start, start + len(sentence)

    stripped_sentence = sentence.strip()
    start = full_text.find(stripped_sentence)
    if start >= 0:
        return start, start + len(stripped_sentence)

    raise ValueError("The target sentence was not found in full_text.")


def repair_sentence_span_with_offsets(
    record: dict[str, Any],
    tokenizer,
) -> None:
    """Retokenize a record and derive an end-exclusive target-sentence span."""
    full_text = record.get("full_text")
    sentence = record.get("sentence")
    if not isinstance(full_text, str) or not isinstance(sentence, str):
        raise KeyError("Span repair requires string 'full_text' and 'sentence' fields.")

    character_start, character_end = find_sentence_character_span(full_text, sentence)
    encoding = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    offsets = encoding.pop("offset_mapping")
    token_ids = encoding["input_ids"]

    overlapping_indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start
        and token_end > character_start
        and token_start < character_end
    ]
    if not overlapping_indices:
        raise ValueError("No tokenizer offsets overlap the target sentence.")

    sentence_start = min(overlapping_indices)
    sentence_end = max(overlapping_indices) + 1
    if sentence_start >= sentence_end:
        raise ValueError("The repaired sentence span is empty.")

    record["full_text_ids"] = [int(token_id) for token_id in token_ids]
    record["sentence_token_start"] = int(sentence_start)
    record["sentence_token_end"] = int(sentence_end)


def validate_stored_span(record: dict[str, Any]) -> None:
    required = ("full_text_ids", "sentence_token_start", "sentence_token_end")
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"Missing required processed fields: {missing}")
    token_ids = record["full_text_ids"]
    start = int(record["sentence_token_start"])
    end = int(record["sentence_token_end"])
    if not 0 <= start < end <= len(token_ids):
        raise ValueError(
            f"Invalid stored sentence span [{start}, {end}) for "
            f"sequence length {len(token_ids)}."
        )


def prepare_records(
    records: list[dict[str, Any]],
    tokenizer,
    *,
    repair_spans: bool,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            if repair_spans:
                repair_sentence_span_with_offsets(record, tokenizer)
            else:
                validate_stored_span(record)
        except Exception as error:
            failures.append(
                {
                    "id": record.get("id", record.get("idx")),
                    "case": record.get("case"),
                    "error": str(error),
                }
            )

    if failures:
        preview = json.dumps(failures[:10], ensure_ascii=False, indent=2)
        raise ValueError(
            f"Could not prepare {len(failures)} records. Examples:\n{preview}"
        )
    return records


def build_aligned_pairs(
    records: Iterable[dict[str, Any]],
    tokenizer,
    first_case: str,
    second_case: str,
    n_samples: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair conditions by item ID and verify target-sentence alignment."""
    by_case: dict[str, dict[str, dict[str, Any]]] = {
        first_case: {},
        second_case: {},
    }

    for record in records:
        case = normalize_case_label(record.get("case"))
        if case not in by_case:
            continue
        identifier = item_identifier(record)
        if identifier in by_case[case]:
            raise ValueError(f"Duplicate item ID {identifier!r} in case {case!r}.")
        by_case[case][identifier] = record

    first_ids = set(by_case[first_case])
    second_ids = set(by_case[second_case])
    if first_ids != second_ids:
        only_first = sorted(first_ids - second_ids)[:10]
        only_second = sorted(second_ids - first_ids)[:10]
        raise ValueError(
            "Condition item IDs do not match. "
            f"Only in {first_case}: {only_first}; only in {second_case}: {only_second}."
        )

    item_ids = sorted(first_ids)
    if n_samples is not None:
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        item_ids = item_ids[:n_samples]

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    text_failures: list[dict[str, Any]] = []
    for identifier in item_ids:
        first = by_case[first_case][identifier]
        second = by_case[second_case][identifier]

        first_span = first["full_text_ids"][
            int(first["sentence_token_start"]) : int(first["sentence_token_end"])
        ]
        second_span = second["full_text_ids"][
            int(second["sentence_token_start"]) : int(second["sentence_token_end"])
        ]
        first_text = tokenizer.decode(first_span, skip_special_tokens=True).strip()
        second_text = tokenizer.decode(second_span, skip_special_tokens=True).strip()

        if normalize_text_for_alignment(first_text) != normalize_text_for_alignment(
            second_text
        ):
            text_failures.append(
                {
                    "id": identifier,
                    "first_decoded": first_text,
                    "second_decoded": second_text,
                }
            )
        pairs.append((first, second))

    if text_failures:
        preview = json.dumps(text_failures[:10], ensure_ascii=False, indent=2)
        raise ValueError(
            f"{len(text_failures)} condition pairs have different target "
            f"sentences after tokenization. Examples:\n{preview}"
        )

    return pairs


def convert_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def load_model_and_tokenizer(args: argparse.Namespace):
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    token = os.getenv("HF_TOKEN")
    common_kwargs = {
        "token": token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.cache_dir is not None:
        common_kwargs["cache_dir"] = str(args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        **common_kwargs,
    )
    if not tokenizer.is_fast and not args.skip_span_repair:
        raise ValueError("Offset-based sentence-span repair requires a fast tokenizer.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=convert_torch_dtype(args.torch_dtype),
        **common_kwargs,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def select_layers(model, requested_layers: list[int] | None) -> list[int]:
    number_of_layers = len(get_transformer_layers(model))
    if requested_layers is None:
        return list(range(number_of_layers))

    selected = sorted(set(requested_layers))
    invalid = [layer for layer in selected if layer < 0 or layer >= number_of_layers]
    if invalid:
        raise ValueError(
            f"Invalid decoder layers {invalid}; valid indices are "
            f"0 through {number_of_layers - 1}."
        )
    return selected


def save_repaired_records(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)


def run_patching(
    model,
    tokenizer,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    layers: list[int],
    first_case: str,
    second_case: str,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    total = len(pairs) * len(layers) * 2

    with tqdm(total=total, desc="Bidirectional activation patching") as progress:
        for first, second in pairs:
            for layer in layers:
                forward = run_single_patch(model, tokenizer, first, second, layer)
                forward["direction"] = f"case{first_case}_to_case{second_case}"
                results.append(forward)
                progress.update(1)

                reverse = run_single_patch(model, tokenizer, second, first, layer)
                reverse["direction"] = f"case{second_case}_to_case{first_case}"
                results.append(reverse)
                progress.update(1)

    return pd.DataFrame(results)


def run_span_mean_baseline(
    model,
    tokenizer,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    layers: list[int],
    first_case: str,
    second_case: str,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    total = len(pairs) * len(layers) * 2

    with tqdm(total=total, desc="Span-mean replacement baseline") as progress:
        for first, second in pairs:
            for layer in layers:
                first_result = run_single_span_mean_replacement(
                    model,
                    tokenizer,
                    first,
                    layer,
                )
                first_result["direction"] = f"span_mean_case{first_case}"
                results.append(first_result)
                progress.update(1)

                second_result = run_single_span_mean_replacement(
                    model,
                    tokenizer,
                    second,
                    layer,
                )
                second_result["direction"] = f"span_mean_case{second_case}"
                results.append(second_result)
                progress.update(1)

    return pd.DataFrame(results)


def save_layer_statistics(results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    statistics = (
        results.groupby(["layer_idx", "direction"])["delta_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    statistics.to_csv(output_path, index=False)
    return statistics


def plot_results(
    patching_results: pd.DataFrame,
    span_mean_results: pd.DataFrame | None,
    output_path: Path,
) -> None:
    patch_curves = (
        patching_results.groupby(["layer_idx", "direction"])["delta_mean"]
        .mean()
        .unstack()
        .sort_index()
    )

    if span_mean_results is None or span_mean_results.empty:
        figure, axis = plt.subplots(figsize=(12, 5))
        for direction in patch_curves.columns:
            axis.plot(
                patch_curves.index,
                patch_curves[direction],
                marker="o",
                label=direction,
            )
        axis.axhline(0, linestyle="--", linewidth=1, alpha=0.6)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Change in mean surprisal (bits)")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return

    span_curve = (
        span_mean_results.groupby("layer_idx")["delta_mean"].mean().sort_index()
    )
    figure, (top_axis, bottom_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.06},
    )

    for direction in patch_curves.columns:
        bottom_axis.plot(
            patch_curves.index,
            patch_curves[direction],
            marker="o",
            label=direction,
        )
    top_axis.plot(
        span_curve.index,
        span_curve.values,
        marker="x",
        linestyle="-.",
        label="Span-mean replacement baseline",
    )

    bottom_axis.axhline(0, linestyle="--", linewidth=1, alpha=0.6)
    bottom_axis.set_xlabel("Layer")
    bottom_axis.set_ylabel("Patching effect (bits)")
    top_axis.set_ylabel("Baseline effect (bits)")
    bottom_axis.grid(alpha=0.3)
    top_axis.grid(alpha=0.3)
    bottom_axis.legend()
    top_axis.legend()

    top_axis.spines["bottom"].set_visible(False)
    bottom_axis.spines["top"].set_visible(False)
    top_axis.tick_params(labeltop=False)
    bottom_axis.xaxis.tick_bottom()

    diagonal_size = 0.012
    top_kwargs = dict(transform=top_axis.transAxes, color="k", clip_on=False)
    top_axis.plot((-diagonal_size, +diagonal_size), (-diagonal_size, +diagonal_size), **top_kwargs)
    top_axis.plot((1 - diagonal_size, 1 + diagonal_size), (-diagonal_size, +diagonal_size), **top_kwargs)
    bottom_kwargs = dict(transform=bottom_axis.transAxes, color="k", clip_on=False)
    bottom_axis.plot((-diagonal_size, +diagonal_size), (1 - diagonal_size, 1 + diagonal_size), **bottom_kwargs)
    bottom_axis.plot((1 - diagonal_size, 1 + diagonal_size), (1 - diagonal_size, 1 + diagonal_size), **bottom_kwargs)

    figure.suptitle("Experiment 2B: Residual-Stream Activation Patching")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args)
    layers = select_layers(model, args.layers)
    records = load_records(args.input_data)
    records = prepare_records(
        records,
        tokenizer,
        repair_spans=not args.skip_span_repair,
    )

    repaired_path = args.output_dir / "processed_data_repaired_spans.json"
    save_repaired_records(records, repaired_path)

    first_case = normalize_case_label(args.source_case)
    second_case = normalize_case_label(args.target_case)
    pairs = build_aligned_pairs(
        records,
        tokenizer,
        first_case,
        second_case,
        args.n_samples,
    )

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Aligned pairs: {len(pairs)}")
    print(f"Decoder layers: {layers}")

    patching_results = run_patching(
        model,
        tokenizer,
        pairs,
        layers,
        first_case,
        second_case,
    )
    patching_path = args.output_dir / "patching_results.csv"
    patching_results.to_csv(patching_path, index=False)

    statistics_path = args.output_dir / "layer_stats.csv"
    save_layer_statistics(patching_results, statistics_path)

    span_mean_results: pd.DataFrame | None = None
    if not args.skip_span_mean_baseline:
        span_mean_results = run_span_mean_baseline(
            model,
            tokenizer,
            pairs,
            layers,
            first_case,
            second_case,
        )
        span_mean_results.to_csv(
            args.output_dir / "span_mean_replacement_results.csv",
            index=False,
        )

    if not args.skip_plot:
        plot_results(
            patching_results,
            span_mean_results,
            args.output_dir / "activation_patching_results.png",
        )

    print(f"Saved repaired data: {repaired_path}")
    print(f"Saved patching results: {patching_path}")
    print(f"Saved layer statistics: {statistics_path}")
    if span_mean_results is not None:
        print(
            "Saved span-mean baseline: "
            f"{args.output_dir / 'span_mean_replacement_results.csv'}"
        )


if __name__ == "__main__":
    main()
