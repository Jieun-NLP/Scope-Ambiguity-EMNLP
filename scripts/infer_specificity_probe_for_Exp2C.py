#!/usr/bin/env python3
"""Apply the trained specificity edge probe to SCOPEX for Experiment 2C.

The specificity probe is trained only on the independent SPECIFICITY dataset.
This script freezes that trained probe and applies it to the second scopal item
(QP2) in SCOPEX without any further optimization.

Pipeline
--------
1. Load SCOPEX records.
2. Identify the target QP2 span with :mod:`src.target_span`.
3. Extract mean-pooled target-NP representations at every hidden-state layer.
4. Stack the layer-wise vectors into [1, L, H].
5. Apply the frozen scalar-mixing edge probe.
6. Save the continuous specificity probability for each SCOPEX instance.

The validation threshold stored in the probe checkpoint is reported for
descriptive classification summaries only. The continuous probability is the
primary specificity score used for downstream analysis, matching the paper.

SCOPEX case convention
----------------------
0: S     | C_spec
1: S     | C_non
2: S_lex | C_spec
3: S_lex | C_non
4: S_pass| C_spec
5: S_pass| C_non

Expected specificity labels used only for descriptive summaries:
0, 2, 4 -> specific (1)
1, 3, 5 -> non-specific (0)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_representation import (  # noqa: E402
    extract_mean_pooled_span_representation,
    load_json_records,
    load_model_and_tokenizer,
)
from src.probing import (  # noqa: E402
    load_probe_checkpoint,
    predict_specificity,
)
from src.target_span import (  # noqa: E402
    get_scopex_target_np_token_span,
    get_target_np_token_span,
    load_target_np_lexicon,
)


CASE_NAMES = {
    0: "S|C_spec",
    1: "S|C_non",
    2: "S_lex|C_spec",
    3: "S_lex|C_non",
    4: "S_pass|C_spec",
    5: "S_pass|C_non",
}

EXPECTED_SPECIFICITY = {
    0: 1,
    1: 0,
    2: 1,
    3: 0,
    4: 1,
    5: 0,
}

# Pairwise comparisons reported in the main Experiment 2C analysis.
PAIRWISE_COMPARISONS = [
    (0, 1, "S|C_spec vs S|C_non"),
    (2, 1, "S_lex|C_spec vs S|C_non"),
    (4, 1, "S_pass|C_spec vs S|C_non"),
    (0, 3, "S|C_spec vs S_lex|C_non"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a trained scalar-mixing specificity edge probe to the "
            "target QP2 spans in SCOPEX."
        )
    )
    parser.add_argument(
        "--scopex-data",
        type=Path,
        required=True,
        help="SCOPEX data in JSON or JSONL format.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Probe checkpoint produced by "
            "train_specificity_probe_for_Exp2C.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for item-level predictions and summary files.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face model identifier used to extract SCOPEX hidden "
            "representations. This must correspond to the model used to "
            "create the probe-training features."
        ),
    )
    parser.add_argument(
        "--target-np-lexicon",
        type=Path,
        default=None,
        help=(
            "Optional JSON lexicon mapping SCOPEX item IDs to QP2 entries "
            "with a 'span_text' field. This is required when SCOPEX records "
            "do not already contain 'span_text'."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("causal_lm", "base"),
        default="causal_lm",
        help="Hugging Face model class to load.",
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
        help="Optional dtype for loading the language model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help=(
            "Maximum input length. Default 4096 preserves the original "
            "Experiment 2C extraction setting."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model code when required.",
    )
    parser.add_argument(
        "--strict-span-validation",
        action="store_true",
        help=(
            "Reject a target span when its decoded surface form is not an "
            "exact normalized match to the lexicon target. By default, "
            "flexible matches found by target_span.py are retained and the "
            "validation result is recorded in the output metadata."
        ),
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help=(
            "Skip records that cannot be processed instead of stopping. "
            "Skipped records are written to inference_summary.json."
        ),
    )
    return parser.parse_args()


def convert_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def get_item_id(record: dict[str, Any], fallback_index: int) -> str:
    identifier = record.get("id", record.get("idx"))
    return str(fallback_index if identifier is None else identifier)


def get_case(record: dict[str, Any]) -> int:
    if record.get("case") is None:
        raise KeyError("SCOPEX record is missing the 'case' field.")
    case = int(record["case"])
    if case not in CASE_NAMES:
        raise ValueError(f"Unexpected SCOPEX case: {case}")
    return case


def resolve_target_span(
    record: dict[str, Any],
    *,
    tokenizer,
    target_np_lexicon,
) -> dict[str, Any]:
    """Resolve QP2 through the public target_span.py API.

    If a record already carries ``span_text``, that annotation is used
    directly. Otherwise the QP2 text is retrieved from the external lexicon.
    """
    full_text = record.get("full_text")
    if not isinstance(full_text, str) or not full_text:
        raise ValueError("SCOPEX record is missing a non-empty 'full_text'.")

    case = get_case(record)
    inline_target = record.get("span_text")

    if isinstance(inline_target, str) and inline_target.strip():
        span_info = get_target_np_token_span(
            full_text=full_text,
            target_np=inline_target,
            tokenizer=tokenizer,
            case_number=case,
            occurrence=-1,
            add_special_tokens=True,
            case_sensitive=False,
            normalize_whitespace=True,
            normalize_quotes=True,
        )
        span_info["source"] = "record.span_text"
        return span_info

    if target_np_lexicon is None:
        raise ValueError(
            "This SCOPEX record has no 'span_text'. Provide "
            "--target-np-lexicon so the QP2 target can be resolved."
        )

    span_info = get_scopex_target_np_token_span(
        record,
        tokenizer,
        target_np_lexicon,
        occurrence=-1,
        add_special_tokens=True,
        case_sensitive=False,
        normalize_whitespace=True,
        normalize_quotes=True,
    )
    span_info["source"] = "target_np_lexicon"
    return span_info


def tokenize_model_input(
    full_text: str,
    tokenizer,
    *,
    max_length: int,
):
    """Tokenize the model input using the same special-token convention."""
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    return encoded["input_ids"], encoded.get("attention_mask")


def extract_scopex_features(
    *,
    model,
    input_ids,
    attention_mask,
    token_span: tuple[int, int],
    layers: list[int],
    device,
) -> np.ndarray:
    """Return one SCOPEX item as a [1, L, H] probe input tensor."""
    start, end = token_span
    sequence_length = int(input_ids.shape[1])

    if start < 0 or end <= start or end > sequence_length:
        raise ValueError(
            f"Target token span {token_span} is outside model input length "
            f"{sequence_length}. The target may have been truncated."
        )

    layer_vectors = extract_mean_pooled_span_representation(
        model,
        input_ids,
        span_start=start,
        span_end=end,
        layers=layers,
        attention_mask=attention_mask,
        device=device,
        output_dtype=np.float32,
    )

    stacked = np.stack(
        [np.asarray(layer_vectors[layer]).reshape(-1) for layer in layers],
        axis=0,
    )
    return stacked[None, :, :]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def summarize_cases(
    results: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[int(result["case"])].append(result)

    summaries: list[dict[str, Any]] = []
    for case in sorted(grouped):
        items = grouped[case]
        probabilities = np.asarray(
            [item["specificity_probability"] for item in items],
            dtype=float,
        )
        predictions = (probabilities >= threshold).astype(int)
        expected = EXPECTED_SPECIFICITY.get(case)

        summary = {
            "case": case,
            "condition": CASE_NAMES[case],
            "n": len(items),
            "expected_specificity": expected,
            "mean_probability": float(probabilities.mean()),
            "std_probability": float(probabilities.std()),
            "median_probability": float(np.median(probabilities)),
            "predicted_specific_rate": float(predictions.mean()),
        }

        if expected is not None:
            gold = np.full(len(items), expected, dtype=int)
            summary["accuracy"] = float(accuracy_score(gold, predictions))

        summaries.append(summary)

    return summaries


def compute_pairwise_metrics(
    results: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate continuous probe scores for the paper's designed case pairs.

    The first case in each pair is treated as the positive (specific) class
    and the second as the negative (non-specific) class. No classifier or
    threshold is re-fit on SCOPEX.
    """
    by_case: dict[int, list[float]] = defaultdict(list)
    for item in results:
        by_case[int(item["case"])].append(
            float(item["specificity_probability"])
        )

    pair_results: list[dict[str, Any]] = []
    for positive_case, negative_case, name in PAIRWISE_COMPARISONS:
        positive_scores = by_case.get(positive_case, [])
        negative_scores = by_case.get(negative_case, [])

        if not positive_scores or not negative_scores:
            continue

        scores = np.asarray(
            positive_scores + negative_scores,
            dtype=float,
        )
        labels = np.asarray(
            [1] * len(positive_scores) + [0] * len(negative_scores),
            dtype=int,
        )
        predictions = (scores >= threshold).astype(int)

        pair_results.append(
            {
                "comparison": name,
                "positive_case": positive_case,
                "negative_case": negative_case,
                "n_positive": len(positive_scores),
                "n_negative": len(negative_scores),
                "pr_auc": float(average_precision_score(labels, scores)),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "accuracy": float(accuracy_score(labels, predictions)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "threshold": float(threshold),
            }
        )

    return pair_results


def save_case_summary_csv(
    path: Path,
    case_summaries: list[dict[str, Any]],
) -> None:
    if not case_summaries:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(case_summaries[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_summaries)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trained specificity probe...")
    probe, checkpoint = load_probe_checkpoint(
        args.checkpoint,
        device=args.device,
    )

    threshold = float(checkpoint.get("metrics", {}).get("threshold", 0.5))
    config = checkpoint["config"]
    probe_num_layers = int(config["num_layers"])
    probe_hidden_dim = int(config["hidden_dim"])

    print(f"Validation threshold: {threshold:.6f}")
    print(
        f"Probe input shape: [N, {probe_num_layers}, {probe_hidden_dim}]"
    )

    print("Loading language model and tokenizer...")
    language_model, tokenizer, device = load_model_and_tokenizer(
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

    available_hidden_states = int(language_model.config.num_hidden_layers) + 1
    if available_hidden_states != probe_num_layers:
        raise ValueError(
            "Probe/language-model layer mismatch: the checkpoint expects "
            f"{probe_num_layers} hidden states, but {args.model} provides "
            f"{available_hidden_states}. Use the same model family/configuration "
            "that produced the SPECIFICITY training features."
        )

    layers = list(range(probe_num_layers))

    records = load_json_records(args.scopex_data)
    target_np_lexicon = (
        load_target_np_lexicon(args.target_np_lexicon)
        if args.target_np_lexicon is not None
        else None
    )

    print(f"Loaded {len(records)} SCOPEX records.")
    print("Running frozen specificity inference...")

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record_index, record in enumerate(
        tqdm(records, desc="SCOPEX inference"),
        start=1,
    ):
        item_id = get_item_id(record, record_index)

        try:
            case = get_case(record)
            span_info = resolve_target_span(
                record,
                tokenizer=tokenizer,
                target_np_lexicon=target_np_lexicon,
            )

            if args.strict_span_validation and not span_info["decoded_well"]:
                raise ValueError(
                    "Decoded target span does not exactly match the requested "
                    f"QP2: requested={span_info['target_np']!r}, "
                    f"decoded={span_info['decoded_text']!r}."
                )

            input_ids, attention_mask = tokenize_model_input(
                record["full_text"],
                tokenizer,
                max_length=args.max_length,
            )

            token_span = tuple(span_info["token_span"])
            features = extract_scopex_features(
                model=language_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_span=token_span,
                layers=layers,
                device=device,
            )

            if features.shape[2] != probe_hidden_dim:
                raise ValueError(
                    "Probe/language-model hidden-size mismatch: checkpoint "
                    f"expects {probe_hidden_dim}, extracted {features.shape[2]}."
                )

            probabilities, _mixing_weights = predict_specificity(
                probe,
                features,
                device=device,
            )
            probability = float(probabilities[0])
            prediction = int(probability >= threshold)

            results.append(
                {
                    "id": item_id,
                    "case": case,
                    "condition": CASE_NAMES[case],
                    "context": record.get("context", ""),
                    "sentence": record.get("sentence", ""),
                    "full_text": record["full_text"],
                    "target_np": span_info["target_np"],
                    "target_source": span_info.get("source"),
                    "char_span_start": int(span_info["char_span"][0]),
                    "char_span_end": int(span_info["char_span"][1]),
                    "token_span_start": int(token_span[0]),
                    "token_span_end": int(token_span[1]),
                    "decoded_target_span": span_info["decoded_text"],
                    "decoded_well": bool(span_info["decoded_well"]),
                    "specificity_probability": probability,
                    "predicted_specificity": prediction,
                    "validation_threshold": threshold,
                    "expected_specificity": EXPECTED_SPECIFICITY.get(case),
                }
            )

        except Exception as error:
            if not args.allow_skipped:
                raise RuntimeError(
                    f"Failed to process SCOPEX record {record_index} "
                    f"(id={item_id!r})."
                ) from error

            skipped.append(
                {
                    "record_index": record_index,
                    "id": item_id,
                    "case": record.get("case"),
                    "error": str(error),
                }
            )

    if not results:
        raise RuntimeError("No SCOPEX records were successfully processed.")

    case_summaries = summarize_cases(results, threshold=threshold)
    pairwise_metrics = compute_pairwise_metrics(
        results,
        threshold=threshold,
    )

    item_path = args.output_dir / "scopex_specificity_predictions.jsonl"
    case_json_path = args.output_dir / "case_summary.json"
    case_csv_path = args.output_dir / "case_summary.csv"
    pairwise_path = args.output_dir / "pairwise_metrics.json"
    summary_path = args.output_dir / "inference_summary.json"

    write_jsonl(item_path, results)
    write_json(case_json_path, case_summaries)
    save_case_summary_csv(case_csv_path, case_summaries)
    write_json(pairwise_path, pairwise_metrics)

    inference_summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "scopex_data": str(args.scopex_data),
        "target_np_lexicon": (
            str(args.target_np_lexicon)
            if args.target_np_lexicon is not None
            else None
        ),
        "num_input_records": len(records),
        "num_processed": len(results),
        "num_skipped": len(skipped),
        "validation_threshold": threshold,
        "probe_input": {
            "num_layers": probe_num_layers,
            "hidden_dim": probe_hidden_dim,
        },
        "pooling": "span-mean over SCOPEX QP2",
        "optimization_on_scopex": False,
        "continuous_probability_is_primary_score": True,
        "skipped_records": skipped,
    }
    write_json(summary_path, inference_summary)

    print("\nCase-wise specificity scores")
    print("----------------------------")
    for summary in case_summaries:
        print(
            f"Case {summary['case']} ({summary['condition']}): "
            f"n={summary['n']}, "
            f"mean_p={summary['mean_probability']:.4f}, "
            f"specific_rate={summary['predicted_specific_rate']:.4f}, "
            f"acc={summary.get('accuracy', float('nan')):.4f}"
        )

    if pairwise_metrics:
        print("\nPairwise probe performance")
        print("--------------------------")
        for pair in pairwise_metrics:
            print(
                f"{pair['comparison']}: "
                f"PR-AUC={pair['pr_auc']:.4f}, "
                f"ROC-AUC={pair['roc_auc']:.4f}, "
                f"Acc={pair['accuracy']:.4f}, "
                f"F1={pair['f1']:.4f}"
            )

    print(f"\nItem-level predictions: {item_path}")
    print(f"Case summary: {case_csv_path}")
    print(f"Pairwise metrics: {pairwise_path}")
    print(f"Inference summary: {summary_path}")
    print(
        "\nInference complete. The probe and validation threshold were frozen; "
        "no parameter or threshold was optimized on SCOPEX."
    )


if __name__ == "__main__":
    main()
