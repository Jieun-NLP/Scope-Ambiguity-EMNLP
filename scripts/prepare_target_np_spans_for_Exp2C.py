#!/usr/bin/env python3
"""Prepare QP2 target-NP spans for Experiment 2C.

This script recreates the target-NP span annotations used for specificity
probing on SCOPEX.  It is adapted from the original
``Extract_qp2_token_ids_for_specific_model_.ipynb`` pipeline, while delegating
the reusable matching/token-span logic to :mod:`src.target_span`.

Why this preprocessing step exists
-----------------------------------
Experiment 2C probes only the second scopal item (QP2), not the full target
sentence.  The same underlying QP2 can have a different surface realization
in lexicalized SCOPEX conditions (cases 2 and 3), e.g.::

    underlying QP2:  "a donkey"
    surface span:    "this donkey"

For that reason, this script stores both:

``qp2_lexical_form``
    The QP2 form supplied by the external lexicon.

``span_text``
    The actual surface string matched in the current SCOPEX record.

Character spans are shared properties of the text, whereas token spans are
tokenizer-dependent.  Run this script separately for each model/tokenizer
(LLaMA, Mistral, Qwen, ...).

Output span convention
----------------------
All spans are end-exclusive Python slices:

    full_text[span_char_start:span_char_end]
    input_ids[span_token_start:span_token_end]

The output records preserve all existing input fields and overwrite/add the
following Exp. 2C fields:

    qp2_lexical_form
    span_text
    span_char_start
    span_char_end
    span_token_start
    span_token_end
    decoded_span_text
    decoded_well

A summary JSON and an error JSONL are also written next to the output file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.target_span import (  # noqa: E402
    char_span_to_token_span,
    find_target_np_char_span,
    load_target_np_lexicon,
    tokenize_with_offsets,
    validate_token_span,
)


SPAN_FIELDS = (
    "span_text",
    "span_char_start",
    "span_char_end",
    "span_token_start",
    "span_token_end",
    "decoded_span_text",
    "decoded_well",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute tokenizer-specific QP2 target-NP spans for SCOPEX "
            "Experiment 2C."
        )
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Current SCOPEX records in JSON or JSONL format.",
    )
    parser.add_argument(
        "--qp2-lexicon",
        type=Path,
        required=True,
        help=(
            "JSON object mapping SCOPEX item IDs to entries containing "
            "the underlying QP2 under the field 'span_text'."
        ),
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        required=True,
        help="Output JSON or JSONL file with refreshed Exp. 2C span fields.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face model/tokenizer identifier, e.g. "
            "meta-llama/Meta-Llama-3-8B."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face tokenizer code when required.",
    )
    parser.add_argument(
        "--occurrence",
        type=int,
        default=-1,
        help=(
            "Which matching occurrence to use when the QP2 appears multiple "
            "times. -1 selects the last occurrence, matching the original "
            "notebook."
        ),
    )
    parser.add_argument(
        "--span-mode",
        choices=("overlap", "inside", "snap_out"),
        default="snap_out",
        help=(
            "Character-to-token mapping policy. The original notebook used "
            "'snap_out' in its main pipeline."
        ),
    )
    parser.add_argument(
        "--special-token-policy",
        choices=("auto", "add", "none"),
        default="auto",
        help=(
            "How to add tokenizer special tokens. 'auto' avoids adding an "
            "extra BOS when full_text already begins with tokenizer.bos_token; "
            "'add' always requests special tokens; 'none' never adds them."
        ),
    )
    parser.add_argument(
        "--strict-surface-validation",
        action="store_true",
        help=(
            "Treat a decoded span that does not match the actual matched "
            "surface string as an unresolved record."
        ),
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help=(
            "Exit successfully even if some records cannot be resolved. "
            "Errors are still logged and unresolved span fields are set to null."
        ),
    )
    return parser.parse_args()


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load JSON/JSONL records and return records plus detected format."""
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number} in {path}."
                    ) from error
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Expected JSON object at line {line_number}; "
                        f"found {type(item).__name__}."
                    )
                records.append(item)
        return records, "jsonl"

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)

        if isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise TypeError("JSON list must contain only objects.")
            return data, "json"

        # Some historical preprocessing files may wrap the records in a key.
        if isinstance(data, dict):
            for key in ("records", "data", "items"):
                value = data.get(key)
                if isinstance(value, list) and all(
                    isinstance(item, dict) for item in value
                ):
                    return value, "json"

        raise TypeError(
            "JSON input must be a list of records or contain a "
            "'records'/'data'/'items' list."
        )

    raise ValueError("Input data must use .json or .jsonl.")


def save_records(
    records: Iterable[dict[str, Any]],
    path: Path,
) -> None:
    """Save records using the format implied by the output suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
        return

    if suffix == ".json":
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(
                list(records),
                output_file,
                ensure_ascii=False,
                indent=2,
            )
        return

    raise ValueError("Output data must use .json or .jsonl.")


def choose_add_special_tokens(
    full_text: str,
    tokenizer,
    policy: str,
) -> bool:
    """Resolve the special-token policy for one record."""
    if policy == "add":
        return True
    if policy == "none":
        return False

    # Auto mode: historical LLaMA files sometimes already contain a literal
    # '<|begin_of_text|>' prefix. Adding another BOS would shift token indices.
    bos_token = getattr(tokenizer, "bos_token", None)
    if isinstance(bos_token, str) and bos_token and full_text.startswith(bos_token):
        return False

    return True


def clear_span_fields(record: dict[str, Any]) -> None:
    """Set the public Exp. 2C span fields to null after a failed resolution."""
    record["qp2_lexical_form"] = record.get("qp2_lexical_form")
    for field in SPAN_FIELDS:
        record[field] = None


def resolve_record_span(
    record: dict[str, Any],
    *,
    qp2_lexicon: dict[str, dict[str, Any]],
    tokenizer,
    occurrence: int,
    span_mode: str,
    special_token_policy: str,
) -> dict[str, Any]:
    """Resolve one SCOPEX QP2 and return refreshed span metadata."""
    raw_id = record.get("id", record.get("idx"))
    if raw_id is None:
        raise KeyError("Record has neither 'id' nor 'idx'.")

    item_id = str(raw_id)
    if item_id not in qp2_lexicon:
        raise KeyError(f"Item ID {item_id!r} is missing from the QP2 lexicon.")

    lexicon_entry = qp2_lexicon[item_id]
    qp2 = lexicon_entry.get("span_text")
    if not isinstance(qp2, str) or not qp2.strip():
        raise ValueError(
            f"QP2 lexicon entry {item_id!r} has no non-empty 'span_text'."
        )

    full_text = record.get("full_text")
    if not isinstance(full_text, str) or not full_text:
        raise ValueError(f"Record {item_id!r} has no non-empty 'full_text'.")

    case_value = record.get("case")
    case_number = int(case_value) if case_value is not None else None

    # Preserve the original notebook's robust matching sequence, including
    # demonstrative alternation specifically for cases 2 and 3.
    char_start, char_end = find_target_np_char_span(
        full_text,
        qp2,
        case_sensitive=False,
        normalize_whitespace=True,
        normalize_quotes=True,
        occurrence=occurrence,
        try_nfkc=True,
        strip_zero_width=True,
        case_number=case_number,
        demo_swap_cases=(2, 3),
    )

    # Store the actual text in this condition, rather than silently keeping
    # the underlying lexicon form after a lexicalized match.
    surface_span = full_text[char_start:char_end]

    add_special_tokens = choose_add_special_tokens(
        full_text,
        tokenizer,
        special_token_policy,
    )
    encoding, offsets = tokenize_with_offsets(
        full_text,
        tokenizer,
        add_special_tokens=add_special_tokens,
    )

    _, snapped_char_span, token_span = char_span_to_token_span(
        full_text,
        (char_start, char_end),
        offsets,
        mode=span_mode,
        trim_whitespace=True,
    )
    if token_span is None:
        raise ValueError(
            f"No tokenizer tokens overlap matched QP2 surface span "
            f"{surface_span!r}."
        )

    # Validate against the *actual surface realization*.  Comparing against
    # the underlying lexicon form would incorrectly mark legitimate case-2/3
    # alternations such as 'a donkey' -> 'this donkey' as failures.
    validation = validate_token_span(
        encoding=encoding,
        tokenizer=tokenizer,
        target_np=surface_span,
        token_span=token_span,
        case_sensitive=False,
        normalize_whitespace=True,
        normalize_quotes=True,
    )

    return {
        "qp2_lexical_form": qp2,
        "span_text": surface_span,
        "span_char_start": int(char_start),
        "span_char_end": int(char_end),
        "span_token_start": int(token_span[0]),
        "span_token_end": int(token_span[1]),
        "decoded_span_text": validation["decoded_text"],
        "decoded_well": bool(validation["decoded_well"]),
        "snapped_char_span": [
            int(snapped_char_span[0]),
            int(snapped_char_span[1]),
        ],
        "add_special_tokens": add_special_tokens,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def companion_paths(output_data: Path) -> tuple[Path, Path]:
    stem = output_data.stem
    return (
        output_data.with_name(f"{stem}_span_summary.json"),
        output_data.with_name(f"{stem}_span_errors.jsonl"),
    )


def main() -> None:
    args = parse_args()

    records, _input_format = load_records(args.input_data)
    qp2_lexicon = load_target_np_lexicon(args.qp2_lexicon)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Target-span preparation requires a fast tokenizer with "
            "offset mappings."
        )

    output_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    resolved_by_case: Counter[int | str] = Counter()
    unresolved_by_case: Counter[int | str] = Counter()
    flexible_surface_changes = 0
    decoded_well_count = 0

    for record_index, source_record in enumerate(
        tqdm(records, desc="Preparing QP2 target spans"),
        start=1,
    ):
        record = dict(source_record)
        raw_id = record.get("id", record.get("idx"))
        item_id = str(record_index if raw_id is None else raw_id)
        case = record.get("case", "NA")

        try:
            span_info = resolve_record_span(
                record,
                qp2_lexicon=qp2_lexicon,
                tokenizer=tokenizer,
                occurrence=args.occurrence,
                span_mode=args.span_mode,
                special_token_policy=args.special_token_policy,
            )

            if (
                args.strict_surface_validation
                and not span_info["decoded_well"]
            ):
                raise ValueError(
                    "Decoded token span does not match the matched surface "
                    f"QP2: surface={span_info['span_text']!r}, "
                    f"decoded={span_info['decoded_span_text']!r}."
                )

            qp2 = span_info["qp2_lexical_form"]
            surface = span_info["span_text"]
            if qp2.strip().lower() != surface.strip().lower():
                flexible_surface_changes += 1

            if span_info["decoded_well"]:
                decoded_well_count += 1

            # Keep the historical public field names used downstream.
            record.update(
                {
                    "qp2_lexical_form": span_info["qp2_lexical_form"],
                    "span_text": span_info["span_text"],
                    "span_char_start": span_info["span_char_start"],
                    "span_char_end": span_info["span_char_end"],
                    "span_token_start": span_info["span_token_start"],
                    "span_token_end": span_info["span_token_end"],
                    "decoded_span_text": span_info["decoded_span_text"],
                    "decoded_well": span_info["decoded_well"],
                }
            )

            resolved_by_case[case] += 1

        except Exception as error:
            clear_span_fields(record)
            unresolved_by_case[case] += 1
            errors.append(
                {
                    "record_index": record_index,
                    "id": item_id,
                    "case": record.get("case"),
                    "reason": type(error).__name__,
                    "message": str(error),
                    "full_text": record.get("full_text", "")[:500],
                }
            )

        output_records.append(record)

    save_records(output_records, args.output_data)

    summary_path, error_path = companion_paths(args.output_data)
    if errors:
        write_jsonl(error_path, errors)
    elif error_path.exists():
        error_path.unlink()

    summary = {
        "input_data": str(args.input_data),
        "qp2_lexicon": str(args.qp2_lexicon),
        "output_data": str(args.output_data),
        "tokenizer": args.model,
        "span_convention": "end-exclusive [start:end)",
        "span_mode": args.span_mode,
        "occurrence": args.occurrence,
        "special_token_policy": args.special_token_policy,
        "num_input_records": len(records),
        "num_resolved": len(records) - len(errors),
        "num_unresolved": len(errors),
        "decoded_well": decoded_well_count,
        "surface_form_differs_from_qp2_lexicon": flexible_surface_changes,
        "resolved_by_case": {
            str(key): value
            for key, value in sorted(
                resolved_by_case.items(),
                key=lambda item: str(item[0]),
            )
        },
        "unresolved_by_case": {
            str(key): value
            for key, value in sorted(
                unresolved_by_case.items(),
                key=lambda item: str(item[0]),
            )
        },
        "error_log": str(error_path) if errors else None,
    }
    write_json(summary_path, summary)

    print(f"Input records: {len(records)}")
    print(f"Resolved spans: {summary['num_resolved']}")
    print(f"Unresolved spans: {summary['num_unresolved']}")
    print(
        "Surface forms differing from lexicon QP2: "
        f"{flexible_surface_changes}"
    )
    print(f"Decoded well: {decoded_well_count}/{len(records)}")
    print(f"Output data: {args.output_data}")
    print(f"Summary: {summary_path}")
    if errors:
        print(f"Error log: {error_path}")

    if errors and not args.allow_unresolved:
        raise SystemExit(
            f"{len(errors)} target spans remain unresolved. "
            "Inspect the error log or rerun with --allow-unresolved "
            "for diagnostic output."
        )


if __name__ == "__main__":
    main()
