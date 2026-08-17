#!/usr/bin/env python3
"""
Reproduce the Experiment 1 tables reported in the paper.

This script analyzes the outputs produced by scripts/Experiment1.py.

Expected input files in --results-dir
-------------------------------------
- case_aggregate.csv
- surprisal_records.jsonl
- token_deltas_insitu_spacy_all.jsonl

Outputs
-------
- table2_sentence_surprisal.csv
- table2_pairwise_statistics.csv
- table3_scope_sensitivity.csv

Case mapping
------------
0 = (S | C_spec)
1 = (S | C_non)
2 = (S_lex | C_spec)
3 = (S_lex | C_non)
4 = (S_pass | C_spec)
5 = (S_pass | C_non)

Notes
-----
The Table 3 scope-item lexicon and aggregation below are intentionally kept
faithful to the original analysis code used to produce the reported results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CASE_LABELS = {
    "0": "(S | C_spec)",
    "1": "(S | C_non)",
    "2": "(S_lex | C_spec)",
    "3": "(S_lex | C_non)",
    "4": "(S_pass | C_spec)",
    "5": "(S_pass | C_non)",
}

PAIR_LABELS = {
    "0-1": "(S | C_spec) - (S | C_non)",
    "2-3": "(S_lex | C_spec) - (S_lex | C_non)",
    "4-5": "(S_pass | C_spec) - (S_pass | C_non)",
}

CASE_PAIRS = [("0", "1"), ("2", "3"), ("4", "5")]


# Faithful copy of the lexicon used in the original Table 3 analysis.
SCOPE_ITEMS = {
    "quantifiers": [
        "every", "each", "all", "most", "some", "many",
        "few", "several", "both", "half",
    ],
    "indefinites": [
        "a", "an", "some", "any", "someone", "anyone",
        "something", "anything", "one",
    ],
    "negations": [
        "no", "not", "never", "none", "nothing", "nobody",
    ],
    "numerals": [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
    ],
}

ALL_SCOPE_LEMMAS = set(sum(SCOPE_ITEMS.values(), []))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}"
                ) from exc
    return records


def reproduce_table2_sentence_means(case_aggregate_path: Path) -> pd.DataFrame:
    """
    Reproduce the upper part of Table 2:
    mean in-situ surprisal, isolated surprisal, and their difference.
    """
    df = pd.read_csv(case_aggregate_path)
    df["case"] = df["case"].astype(str)

    required = {
        "case",
        "mean_surprisal_insitu",
        "mean_surprisal_isolated",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{case_aggregate_path} is missing columns: {sorted(missing)}"
        )

    out = df[
        ["case", "mean_surprisal_insitu", "mean_surprisal_isolated"]
    ].copy()

    out["condition"] = out["case"].map(CASE_LABELS)
    out["diff"] = (
        out["mean_surprisal_insitu"]
        - out["mean_surprisal_isolated"]
    )

    return out[
        [
            "case",
            "condition",
            "mean_surprisal_insitu",
            "mean_surprisal_isolated",
            "diff",
        ]
    ].sort_values("case")


def reproduce_table2_pairwise_stats(
    surprisal_records_path: Path,
) -> pd.DataFrame:
    """
    Reproduce the pairwise statistical comparisons in Table 2.

    The original analysis compares itemwise contextual facilitation:
        mode_diff_mean = in-situ surprisal - isolated surprisal

    For each paired contrast, this script reports:
    - mean paired difference
    - Wilcoxon signed-rank p-value
    - paired-samples Cohen's dz

    Additional statistics from the original analysis are also retained.
    """
    df = pd.DataFrame(read_jsonl(surprisal_records_path))

    required = {"id", "case", "mode_diff_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{surprisal_records_path} is missing columns: {sorted(missing)}"
        )

    df["id"] = df["id"].astype(str)
    df["case"] = df["case"].astype(str)
    df["mode_diff_mean"] = pd.to_numeric(
        df["mode_diff_mean"], errors="coerce"
    )

    pivot = df.pivot_table(
        index="id",
        columns="case",
        values="mode_diff_mean",
        aggfunc="mean",
    )

    rows = []

    for case_a, case_b in CASE_PAIRS:
        pair = f"{case_a}-{case_b}"
        paired = pivot[[case_a, case_b]].dropna()

        x = paired[case_a].to_numpy(dtype=float)
        y = paired[case_b].to_numpy(dtype=float)
        d = x - y
        n = len(d)

        mean_diff = float(np.mean(d))
        sd_diff = float(np.std(d, ddof=1))
        cohen_dz = mean_diff / sd_diff if sd_diff != 0 else np.nan

        wil = stats.wilcoxon(
            x,
            y,
            zero_method="pratt",
            alternative="two-sided",
        )

        t_res = stats.ttest_rel(x, y, nan_policy="omit")
        shapiro = stats.shapiro(d)

        rows.append(
            {
                "pair": pair,
                "comparison": PAIR_LABELS[pair],
                "n_pairs": n,
                "mean_delta": mean_diff,
                "wilcoxon_W": float(wil.statistic),
                "wilcoxon_p": float(wil.pvalue),
                "cohen_dz": cohen_dz,
                "paired_t": float(t_res.statistic),
                "paired_t_p": float(t_res.pvalue),
                "shapiro_W": float(shapiro.statistic),
                "shapiro_p": float(shapiro.pvalue),
            }
        )

    return pd.DataFrame(rows)


def reproduce_table3_scope_sensitivity(
    token_deltas_path: Path,
) -> pd.DataFrame:
    """
    Reproduce Table 3 from token-level absolute surprisal differences.

    Original analysis logic:
    1. classify each token as scope vs. non-scope from its spaCy lemma;
    2. average abs_delta separately for the two groups within each pair;
    3. report Scope - Non-scope and Scope / Non-scope.
    """
    df = pd.DataFrame(read_jsonl(token_deltas_path))

    required = {"pair", "lemma", "abs_delta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{token_deltas_path} is missing columns: {sorted(missing)}"
        )

    # Keep the original behavior exactly.
    df["is_scope"] = df["lemma"].isin(ALL_SCOPE_LEMMAS)
    df["abs_delta"] = pd.to_numeric(df["abs_delta"], errors="coerce")

    result = (
        df.groupby(["pair", "is_scope"])["abs_delta"]
        .mean()
        .reset_index()
        .pivot(index="pair", columns="is_scope", values="abs_delta")
        .rename(
            columns={
                False: "non_scope_mean",
                True: "scope_mean",
            }
        )
        .reset_index()
    )

    result["comparison"] = result["pair"].map(PAIR_LABELS)
    result["diff"] = result["scope_mean"] - result["non_scope_mean"]
    result["ratio"] = result["scope_mean"] / result["non_scope_mean"]

    order = ["0-1", "2-3", "4-5"]
    result["pair"] = pd.Categorical(
        result["pair"], categories=order, ordered=True
    )

    return (
        result[
            [
                "pair",
                "comparison",
                "non_scope_mean",
                "scope_mean",
                "diff",
                "ratio",
            ]
        ]
        .sort_values("pair")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Experiment 1 Tables 2 and 3 from the outputs "
            "of scripts/Experiment1.py."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing case_aggregate.csv, surprisal_records.jsonl, "
            "and token_deltas_insitu_spacy_all.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <results-dir>/tables",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = (
        args.output_dir if args.output_dir is not None
        else results_dir / "tables"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    case_aggregate_path = results_dir / "case_aggregate.csv"
    surprisal_records_path = results_dir / "surprisal_records.jsonl"
    token_deltas_path = results_dir / "token_deltas_insitu_spacy_all.jsonl"

    for path in (
        case_aggregate_path,
        surprisal_records_path,
        token_deltas_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    table2_means = reproduce_table2_sentence_means(case_aggregate_path)
    table2_stats = reproduce_table2_pairwise_stats(surprisal_records_path)
    table3 = reproduce_table3_scope_sensitivity(token_deltas_path)

    table2_means_path = output_dir / "table2_sentence_surprisal.csv"
    table2_stats_path = output_dir / "table2_pairwise_statistics.csv"
    table3_path = output_dir / "table3_scope_sensitivity.csv"

    table2_means.to_csv(
        table2_means_path, index=False, encoding="utf-8-sig"
    )
    table2_stats.to_csv(
        table2_stats_path, index=False, encoding="utf-8-sig"
    )
    table3.to_csv(
        table3_path, index=False, encoding="utf-8-sig"
    )

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("\n=== Table 2: Mean surprisal ===")
    print(table2_means.to_string(index=False))

    print("\n=== Table 2: Pairwise statistical comparisons ===")
    print(table2_stats.to_string(index=False))

    print("\n=== Table 3: Scope sensitivity ===")
    print(table3.to_string(index=False))

    print("\nSaved:")
    print(f"  {table2_means_path}")
    print(f"  {table2_stats_path}")
    print(f"  {table3_path}")


if __name__ == "__main__":
    main()
