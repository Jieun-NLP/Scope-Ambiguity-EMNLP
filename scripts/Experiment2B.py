#!/usr/bin/env python3
"""
Experiment 2B: activation patching with precomputed, token-aligned sentence spans.

This implementation is designed for the final SCOPEX analysis.

Methodological choices
----------------------
1. The script DOES NOT re-tokenize ``full_text`` and DOES NOT repair sentence
   spans at run time. It uses the token IDs and end-exclusive sentence spans
   created during preprocessing:
       full_text_ids
       sentence_token_start
       sentence_token_end

2. Before patching, the two contextual conditions are aligned by item ID and
   their stored target-sentence token ID sequences are compared EXACTLY.
   Position-wise activation patching is performed only when the token sequences
   are identical. This is essential because the intervention assumes that token
   position j in the source corresponds to token position j in the target.

3. Mismatched pairs are skipped by default and written to
   ``excluded_alignment_mismatches.json``. Use ``--mismatch-policy error`` to
   fail instead.

4. Sentence spans use Python's standard end-exclusive convention [start, end).

5. The intervention is applied at the output of each decoder block (the
   post-MLP residual-stream state for LLaMA/Mistral/Qwen-style decoder blocks).

6. Patching is bidirectional:
       case 0 -> case 1
       case 1 -> case 0
   By default these correspond to:
       (S | C_spec) -> (S | C_non)
       (S | C_non)  -> (S | C_spec)

7. The causal effect is:
       delta_mean = patched mean sentence surprisal - clean mean sentence surprisal

8. The span-mean replacement baseline is retained.

This script intentionally avoids the offset-based re-tokenization/alignment
repair that can introduce new boundary-token mismatches across contexts.
"""

from __future__ import annotations

import argparse
import json
import os
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


CASE_NAMES = {
    "0": "S|C_spec",
    "1": "S|C_non",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2B: bidirectional residual-stream activation patching "
            "using precomputed, exactly aligned target-sentence token spans."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID.")
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Processed JSON/JSONL containing stored token IDs and sentence spans.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for patching results, statistics, diagnostics, and figure.",
    )
    parser.add_argument("--source-case", default="0")
    parser.add_argument("--target-case", default="1")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Decoder layers to test. Omit to use all layers.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Optional number of VALID aligned pairs for a test run.",
    )
    parser.add_argument(
        "--mismatch-policy",
        choices=("skip", "error"),
        default="skip",
        help=(
            "What to do if stored target token IDs differ across conditions. "
            "Default: skip and report the pair."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-span-mean-baseline", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input data file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if suffix == ".json":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("A JSON dataset must contain a top-level list.")
            return data
    raise ValueError("Input data must use .json or .jsonl.")


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


def validate_stored_record(record: dict[str, Any]) -> None:
    required = ("full_text_ids", "sentence_token_start", "sentence_token_end")
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"Missing required processed fields: {missing}")

    token_ids = record["full_text_ids"]
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("'full_text_ids' must be a non-empty list.")

    start = int(record["sentence_token_start"])
    end = int(record["sentence_token_end"])

    # Stored SCOPEX spans are end-exclusive: [start, end).
    if not 0 <= start < end <= len(token_ids):
        raise ValueError(
            f"Invalid end-exclusive sentence span [{start}, {end}) "
            f"for sequence length {len(token_ids)}."
        )


def stored_sentence_ids(record: dict[str, Any]) -> list[int]:
    start = int(record["sentence_token_start"])
    end = int(record["sentence_token_end"])
    return [int(x) for x in record["full_text_ids"][start:end]]


def build_exactly_aligned_pairs(
    records: Iterable[dict[str, Any]],
    tokenizer,
    first_case: str,
    second_case: str,
    mismatch_policy: str,
    n_samples: int | None,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    """
    Align condition pairs by item ID and require exact stored target-token IDs.

    Exact equality is deliberately stronger than decoded-text equality because
    residual-stream patching is position-wise. Equal strings with different
    tokenizations are not considered safe for direct token-position patching.
    """
    by_case: dict[str, dict[str, dict[str, Any]]] = {
        first_case: {},
        second_case: {},
    }

    for record in records:
        case = normalize_case_label(record.get("case"))
        if case not in by_case:
            continue
        validate_stored_record(record)
        identifier = item_identifier(record)
        if identifier in by_case[case]:
            raise ValueError(f"Duplicate item ID {identifier!r} in case {case!r}.")
        by_case[case][identifier] = record

    first_ids = set(by_case[first_case])
    second_ids = set(by_case[second_case])
    if first_ids != second_ids:
        raise ValueError(
            "Condition item IDs do not match. "
            f"Only in {first_case}: {sorted(first_ids-second_ids)[:10]}; "
            f"only in {second_case}: {sorted(second_ids-first_ids)[:10]}."
        )

    valid_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    mismatches: list[dict[str, Any]] = []

    # Numeric sort when possible, otherwise lexical.
    def sort_key(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)

    for identifier in sorted(first_ids, key=sort_key):
        first = by_case[first_case][identifier]
        second = by_case[second_case][identifier]

        ids_first = stored_sentence_ids(first)
        ids_second = stored_sentence_ids(second)

        if ids_first != ids_second:
            mismatches.append(
                {
                    "id": identifier,
                    "first_case": first_case,
                    "second_case": second_case,
                    "first_span_length": len(ids_first),
                    "second_span_length": len(ids_second),
                    "first_token_ids": ids_first,
                    "second_token_ids": ids_second,
                    "first_decoded": tokenizer.decode(
                        ids_first, skip_special_tokens=True
                    ),
                    "second_decoded": tokenizer.decode(
                        ids_second, skip_special_tokens=True
                    ),
                    "first_sentence": first.get("sentence"),
                    "second_sentence": second.get("sentence"),
                }
            )
            continue

        valid_pairs.append((first, second))

    if mismatches and mismatch_policy == "error":
        preview = json.dumps(mismatches[:10], ensure_ascii=False, indent=2)
        raise ValueError(
            f"{len(mismatches)} pair(s) have non-identical stored target-token "
            f"sequences and cannot be position-wise patched safely.\n{preview}"
        )

    if n_samples is not None:
        if n_samples < 1:
            raise ValueError("--n-samples must be positive.")
        valid_pairs = valid_pairs[:n_samples]

    return valid_pairs, mismatches


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
    kwargs = {
        "token": token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        **kwargs,
    )

    # Keep compatibility with current Transformers while avoiding dependence on
    # the deprecated torch_dtype keyword when possible.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=convert_torch_dtype(args.torch_dtype),
        **kwargs,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def select_layers(model, requested_layers: list[int] | None) -> list[int]:
    n_layers = len(get_transformer_layers(model))
    if requested_layers is None:
        return list(range(n_layers))

    selected = sorted(set(requested_layers))
    invalid = [x for x in selected if x < 0 or x >= n_layers]
    if invalid:
        raise ValueError(
            f"Invalid decoder layers {invalid}; valid range is 0..{n_layers-1}."
        )
    return selected


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
                forward["direction_label"] = (
                    f"{CASE_NAMES.get(first_case, 'case'+first_case)} -> "
                    f"{CASE_NAMES.get(second_case, 'case'+second_case)}"
                )
                results.append(forward)
                progress.update(1)

                reverse = run_single_patch(model, tokenizer, second, first, layer)
                reverse["direction"] = f"case{second_case}_to_case{first_case}"
                reverse["direction_label"] = (
                    f"{CASE_NAMES.get(second_case, 'case'+second_case)} -> "
                    f"{CASE_NAMES.get(first_case, 'case'+first_case)}"
                )
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
                    model, tokenizer, first, layer
                )
                first_result["direction"] = f"span_mean_case{first_case}"
                results.append(first_result)
                progress.update(1)

                second_result = run_single_span_mean_replacement(
                    model, tokenizer, second, layer
                )
                second_result["direction"] = f"span_mean_case{second_case}"
                results.append(second_result)
                progress.update(1)

    return pd.DataFrame(results)


def save_layer_statistics(results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    stats_df = (
        results.groupby(["layer_idx", "direction", "direction_label"])["delta_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats_df.to_csv(output_path, index=False)
    return stats_df


def plot_results(
    patching_results: pd.DataFrame,
    span_mean_results: pd.DataFrame | None,
    output_path: Path,
) -> None:
    patch_curves = (
        patching_results.groupby(["layer_idx", "direction_label"])["delta_mean"]
        .mean()
        .unstack()
        .sort_index()
    )

    if span_mean_results is None or span_mean_results.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        for label in patch_curves.columns:
            ax.plot(patch_curves.index, patch_curves[label], marker="o", label=label)
        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Change in mean sentence surprisal (bits)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    span_curve = (
        span_mean_results.groupby("layer_idx")["delta_mean"].mean().sort_index()
    )

    fig, (top_ax, bottom_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.06},
    )
    plot_order = [
    "S|C_spec -> S|C_non",
    "S|C_non -> S|C_spec",
    ]
    
    STYLE_MAP = {"S|C_spec -> S|C_non": {"marker": "o"   # blue + circle
                                         },
        "S|C_non -> S|C_spec": {
        "marker": "^"   # orange + triangle
        },}

    for label in plot_order:
        if label not in patch_curves.columns:
            continue
        
        bottom_ax.plot(patch_curves.index, 
                       patch_curves[label], 
                       marker=STYLE_MAP[label]["marker"], 
                       label=label
        )

    top_ax.plot(
        span_curve.index,
        span_curve.values,
        marker="x",
        linestyle="-.",
        label="Span-mean replacement baseline",
    )

    bottom_ax.axhline(0, linestyle="--", linewidth=1, alpha=0.6)
    bottom_ax.set_xlabel("Layer")
    bottom_ax.set_ylabel("Patching effect (bits)")
    top_ax.set_ylabel("Baseline effect (bits)")
    bottom_ax.grid(alpha=0.3)
    top_ax.grid(alpha=0.3)
    bottom_ax.legend()
    top_ax.legend()

    top_ax.spines["bottom"].set_visible(False)
    bottom_ax.spines["top"].set_visible(False)

    fig.suptitle("Experiment 2B: Residual-Stream Activation Patching")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args)
    layers = select_layers(model, args.layers)
    records = load_records(args.input_data)

    first_case = normalize_case_label(args.source_case)
    second_case = normalize_case_label(args.target_case)

    pairs, mismatches = build_exactly_aligned_pairs(
        records,
        tokenizer,
        first_case,
        second_case,
        args.mismatch_policy,
        args.n_samples,
    )

    mismatch_path = args.output_dir / "excluded_alignment_mismatches.json"
    with mismatch_path.open("w", encoding="utf-8") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=2)

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Candidate paired IDs: {len(pairs) + len(mismatches)}")
    print(f"Exactly token-aligned pairs used: {len(pairs)}")
    print(f"Excluded alignment mismatches: {len(mismatches)}")
    if mismatches:
        print("Excluded IDs:", ", ".join(str(x["id"]) for x in mismatches))
    print(f"Decoder layers: {layers}")

    if not pairs:
        raise ValueError("No exactly aligned condition pairs remain for patching.")

    patching_results = run_patching(
        model, tokenizer, pairs, layers, first_case, second_case
    )
    patching_path = args.output_dir / "patching_results.csv"
    patching_results.to_csv(patching_path, index=False)

    stats_path = args.output_dir / "layer_stats.csv"
    save_layer_statistics(patching_results, stats_path)

    span_mean_results = None
    if not args.skip_span_mean_baseline:
        span_mean_results = run_span_mean_baseline(
            model, tokenizer, pairs, layers, first_case, second_case
        )
        span_mean_results.to_csv(
            args.output_dir / "span_mean_replacement_results.csv", index=False
        )

    if not args.skip_plot:
        plot_results(
            patching_results,
            span_mean_results,
            args.output_dir / "activation_patching_results.png",
        )

    print(f"Saved patching results: {patching_path}")
    print(f"Saved layer statistics: {stats_path}")
    print(f"Saved alignment diagnostics: {mismatch_path}")
    if span_mean_results is not None:
        print(
            "Saved span-mean baseline: "
            f"{args.output_dir / 'span_mean_replacement_results.csv'}"
        )


if __name__ == "__main__":
    main()
