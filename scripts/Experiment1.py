"""Experiment 1: context-dependent sentence surprisal analysis.

The script compares the surprisal of the same target sentence when it is
processed with its preceding context (in-situ) and as an isolated sentence.
It also computes token-level surprisal differences for predefined case pairs
and enriches them with spaCy part-of-speech and lemma annotations.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (
    SurprisalCache,
    SurprisalConfig,
    analyze_token_deltas_for_pairs_spacy,
    compute_record_surprisals,
    load_jsonl,
)


def parse_case_pairs(values: list[str]) -> list[Tuple[str, str]]:
    pairs: list[Tuple[str, str]] = []
    for value in values:
        parts = value.split("-")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid case pair '{value}'. Use the form A-B, for example 0-1."
            )
        pairs.append((parts[0], parts[1]))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for caches and results.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Hugging Face model name or local path.")
    parser.add_argument("--case-pairs", nargs="+", default=["0-1", "2-3", "4-5"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--log-base", type=float, default=2.0)
    parser.add_argument("--skip-token-analysis", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    """Load a causal language model and its matching fast tokenizer."""
    token = os.getenv("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=token,
    )
    model.eval().to(device)
    return model, tokenizer, device


def main() -> None:
    args = parse_args()
    case_pairs = parse_case_pairs(args.case_pairs)
    records = load_jsonl(args.input)
    config = SurprisalConfig(
        output_root=args.output_root,
        log_base=args.log_base,
    )

    model, tokenizer, device = load_model_and_tokenizer(args.model)
    cache = SurprisalCache(args.output_root)

    print(f"Loaded {len(records)} records")
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    if not args.skip_token_analysis:
        token_results = analyze_token_deltas_for_pairs_spacy(
            model,
            tokenizer,
            records,
            config,
            cache,
            case_pairs=case_pairs,
            top_k=args.top_k,
            spacy_model=args.spacy_model,
        )
        print(f"Token-level results: {token_results['raw_jsonl']}")
        

    sentence_results = compute_record_surprisals(
        model,
        tokenizer,
        records,
        config,
        cache,
    )
    print(f"Sentence-level results: {sentence_results['records_jsonl']}")
    print(f"Case summary: {sentence_results['case_csv']}")
    print(f"Records processed: {sentence_results['num_records']}")


if __name__ == "__main__":
    main()
