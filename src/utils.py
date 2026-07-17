"""Shared utilities for Experiment 1 surprisal analysis."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SurprisalConfig:
    """Configuration for sentence- and token-level surprisal computation."""

    output_root: Path
    exclude_special: bool = True
    add_bos_if_available: bool = True
    log_base: float = 2.0

    @property
    def surprisal_dir(self) -> Path:
        return self.output_root / "surprisal"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load non-empty JSON objects from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write dictionaries to a UTF-8 JSONL file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_case(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value)


def special_token_mask(tokenizer: Any, token_ids: Sequence[int]) -> List[int]:
    """Return a mask in which 1 marks a tokenizer special token."""
    if hasattr(tokenizer, "get_special_tokens_mask"):
        return tokenizer.get_special_tokens_mask(
            list(token_ids), already_has_special_tokens=True
        )

    special_ids = {
        int(token_id)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        if (token_id := getattr(tokenizer, name, None)) is not None
    }
    return [int(token_id in special_ids) for token_id in token_ids]


@torch.inference_mode()
def token_logprobs_for_sequence(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return log p(x_t | x_<t) for each token position in one sequence."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape (1, sequence_length)")

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered = torch.gather(
        log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)

    full = torch.full(
        (input_ids.size(1),),
        float("nan"),
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    full[1:] = gathered[0]
    return full.cpu()


def compute_token_surprisal(
    model: Any,
    tokenizer: Any,
    full_text_ids: Sequence[int],
    start: int,
    end: int,
    *,
    mode: str,
    exclude_special: bool = True,
    add_bos_if_available: bool = True,
    log_base: float = 2.0,
) -> Tuple[List[float], List[int], List[str]]:
    """Compute token surprisals for a sentence span in context or isolation."""
    sentence_ids = [int(token_id) for token_id in full_text_ids[start : end + 1]]

    if mode == "insitu":
        sequence_ids = [int(token_id) for token_id in full_text_ids]
        offset = start
    elif mode == "isolated":
        bos_id = getattr(tokenizer, "bos_token_id", None)
        use_bos = add_bos_if_available and bos_id is not None
        sequence_ids = ([int(bos_id)] if use_bos else []) + sentence_ids
        offset = 1 if use_bos else 0
    else:
        raise ValueError("mode must be either 'insitu' or 'isolated'")

    device = next(model.parameters()).device
    input_ids = torch.tensor([sequence_ids], dtype=torch.long, device=device)
    log_probs = token_logprobs_for_sequence(model, input_ids).tolist()
    sentence_log_probs = log_probs[offset : offset + len(sentence_ids)]

    mask = (
        special_token_mask(tokenizer, sentence_ids)
        if exclude_special
        else [0] * len(sentence_ids)
    )

    surprisals: List[float] = []
    kept_ids: List[int] = []
    for log_prob, token_id, is_special in zip(sentence_log_probs, sentence_ids, mask):
        if log_prob is None or math.isnan(float(log_prob)) or is_special:
            continue
        surprisal = -float(log_prob)
        if log_base != math.e:
            surprisal /= math.log(log_base)
        surprisals.append(surprisal)
        kept_ids.append(token_id)

    pieces = [str(piece) for piece in tokenizer.convert_ids_to_tokens(kept_ids)]
    return surprisals, kept_ids, pieces


class SurprisalCache:
    """Cache token-level surprisals in memory and compressed NumPy files."""

    def __init__(self, output_root: Path):
        self.cache_root = output_root / "surprisal" / "cache"
        ensure_dir(self.cache_root)
        self.memory: Dict[Tuple[Any, ...], Tuple[List[float], List[int], List[str]]] = {}

    def _key_and_path(
        self,
        sid: str,
        case: str,
        mode: str,
        start: int,
        end: int,
        config: SurprisalConfig,
    ) -> Tuple[Tuple[Any, ...], Path]:
        key = (
            sid,
            case,
            mode,
            start,
            end,
            config.log_base,
            config.exclude_special,
            config.add_bos_if_available,
        )
        digest = hashlib.md5("|".join(map(str, key)).encode("utf-8")).hexdigest()
        path = self.cache_root / mode / f"{digest}.npz"
        ensure_dir(path.parent)
        return key, path

    def get_or_compute(
        self,
        model: Any,
        tokenizer: Any,
        full_text_ids: Sequence[int],
        start: int,
        end: int,
        *,
        sid: str,
        case: str,
        mode: str,
        config: SurprisalConfig,
    ) -> Tuple[List[float], List[int], List[str]]:
        key, path = self._key_and_path(sid, case, mode, start, end, config)
        if key in self.memory:
            return self.memory[key]

        if path.exists():
            cached = np.load(path, allow_pickle=True)
            result = (
                cached["surprisals"].astype(np.float32).tolist(),
                cached["kept_ids"].astype(np.int64).tolist(),
                cached["pieces"].tolist(),
            )
            self.memory[key] = result
            return result

        result = compute_token_surprisal(
            model,
            tokenizer,
            full_text_ids,
            start,
            end,
            mode=mode,
            exclude_special=config.exclude_special,
            add_bos_if_available=config.add_bos_if_available,
            log_base=config.log_base,
        )
        self.memory[key] = result
        np.savez_compressed(
            path,
            surprisals=np.asarray(result[0], dtype=np.float32),
            kept_ids=np.asarray(result[1], dtype=np.int64),
            pieces=np.asarray(result[2], dtype=object),
        )
        return result


def _load_spacy(model_name: str):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


def _decode_and_align_offsets(tokenizer: Any, token_ids: Sequence[int]):
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for offset alignment.")
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return text, encoded["input_ids"], encoded["offset_mapping"]


def analyze_token_deltas_for_pairs_spacy(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    config: SurprisalConfig,
    cache: SurprisalCache,
    *,
    case_pairs: Sequence[Tuple[str, str]],
    mode: str = "insitu",
    top_k: int = 100,
    spacy_model: str = "en_core_web_sm",
    output_name: str = "token_deltas_insitu_spacy",
) -> Dict[str, Any]:
    """Compute paired token-surprisal differences and attach spaCy annotations."""
    nlp = _load_spacy(spacy_model)
    output_dir = config.surprisal_dir
    ensure_dir(output_dir)

    indexed = {
        (str(record.get("idx", record.get("id"))), normalize_case(record.get("case"))): record
        for record in records
    }
    rows: List[Dict[str, Any]] = []
    pairs_processed = 0

    for sid in sorted({key[0] for key in indexed}):
        for case_a, case_b in case_pairs:
            record_a = indexed.get((sid, case_a))
            record_b = indexed.get((sid, case_b))
            if record_a is None or record_b is None:
                continue

            ids_a = record_a.get("full_text_ids", [])
            ids_b = record_b.get("full_text_ids", [])
            if not ids_a or not ids_b:
                continue

            start_a = int(record_a.get("sentence_token_start", 0))
            end_a = int(record_a.get("sentence_token_end", len(ids_a) - 1))
            start_b = int(record_b.get("sentence_token_start", 0))
            end_b = int(record_b.get("sentence_token_end", len(ids_b) - 1))

            scores_a, kept_a, _ = cache.get_or_compute(
                model, tokenizer, ids_a, start_a, end_a,
                sid=sid, case=case_a, mode=mode, config=config,
            )
            scores_b, kept_b, _ = cache.get_or_compute(
                model, tokenizer, ids_b, start_b, end_b,
                sid=sid, case=case_b, mode=mode, config=config,
            )
            if not scores_a or len(scores_a) != len(scores_b):
                continue

            text, reencoded_ids, offsets = _decode_and_align_offsets(tokenizer, kept_b)
            length = min(len(scores_a), len(scores_b), len(kept_a), len(kept_b), len(offsets))
            if reencoded_ids != kept_b:
                scores_a, scores_b = scores_a[:length], scores_b[:length]
                kept_a, kept_b, offsets = kept_a[:length], kept_b[:length], offsets[:length]

            doc = nlp(text)

            def spacy_index(offset: Tuple[int, int]) -> Optional[int]:
                start, end = offset
                center = start if end <= start else (start + end - 1) // 2
                for index, token in enumerate(doc):
                    if token.idx <= center < token.idx + len(token.text):
                        return index
                if not doc:
                    return None
                return min(range(len(doc)), key=lambda i: abs(center - doc[i].idx))

            pair_label = f"{case_a}-{case_b}"
            pairs_processed += 1
            for index, (score_a, score_b, token_a, token_b, offset) in enumerate(
                zip(scores_a, scores_b, kept_a, kept_b, offsets)
            ):
                token_index = spacy_index(offset)
                token = doc[token_index] if token_index is not None else None
                delta = float(score_b - score_a)
                rows.append({
                    "id": sid,
                    "pair": pair_label,
                    "subword_idx": index,
                    "surprisal_A": float(score_a),
                    "surprisal_B": float(score_b),
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "token_id_A": int(token_a),
                    "token_id_B": int(token_b),
                    "offset_start": int(offset[0]),
                    "offset_end": int(offset[1]),
                    "word": token.text if token else "",
                    "lemma": token.lemma_ if token else "",
                    "pos_tag": token.pos_ if token else "UNK",
                    "fine_tag": token.tag_ if token else "UNK",
                    "mode": mode,
                    "log_base": config.log_base,
                })

    raw_path = output_dir / f"{output_name}_all.jsonl"
    write_jsonl(rows, raw_path)

    if rows:
        frame = pd.DataFrame(rows)
        for pair_label, subset in frame.groupby("pair"):
            top = subset.nlargest(top_k, "abs_delta")
            bottom = subset.nsmallest(top_k, "abs_delta")
            top.to_csv(output_dir / f"{output_name}_{pair_label}_topk.csv", index=False)
            bottom.to_csv(output_dir / f"{output_name}_{pair_label}_bottomk.csv", index=False)

            for label, selected in (("top", top), ("bottom", bottom)):
                counts = selected["pos_tag"].value_counts().rename_axis("pos_tag").reset_index(name="count")
                counts["share"] = counts["count"] / max(1, len(selected))
                counts.to_csv(output_dir / f"{output_name}_{pair_label}_agg_pos_{label}.csv", index=False)

            aggregate = (
                subset.groupby("lemma", dropna=False)
                .agg(mean_abs_delta=("abs_delta", "mean"), mean_delta=("delta", "mean"), n=("abs_delta", "count"))
                .reset_index()
                .sort_values(["mean_abs_delta", "n"], ascending=[False, False])
            )
            aggregate.to_csv(output_dir / f"{output_name}_{pair_label}_agg_word.csv", index=False)

    return {
        "raw_jsonl": str(raw_path),
        "pairs_processed": pairs_processed,
        "output_dir": str(output_dir),
    }


def compute_record_surprisals(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    config: SurprisalConfig,
    cache: SurprisalCache,
) -> Dict[str, Any]:
    """Compute per-record in-situ and isolated sentence surprisal and case summaries."""
    output_rows: List[Dict[str, Any]] = []

    for record in records:
        sid = str(record.get("idx", record.get("id")))
        case = normalize_case(record.get("case"))
        full_ids = record.get("full_text_ids", [])
        if not full_ids:
            continue

        start = max(0, min(int(record.get("sentence_token_start", 0)), len(full_ids) - 1))
        end = max(start, min(int(record.get("sentence_token_end", len(full_ids) - 1)), len(full_ids) - 1))

        insitu, _, _ = cache.get_or_compute(
            model, tokenizer, full_ids, start, end,
            sid=sid, case=case, mode="insitu", config=config,
        )
        isolated, _, _ = cache.get_or_compute(
            model, tokenizer, full_ids, start, end,
            sid=sid, case=case, mode="isolated", config=config,
        )

        insitu_mean = float(np.mean(insitu)) if insitu else float("nan")
        isolated_mean = float(np.mean(isolated)) if isolated else float("nan")
        output_rows.append({
            "id": sid,
            "case": case,
            "sentence_token_start": start,
            "sentence_token_end": end,
            "mode_insitu_mean": insitu_mean,
            "mode_insitu_sum": float(np.sum(insitu)) if insitu else float("nan"),
            "mode_insitu_tokens_used": len(insitu),
            "mode_isolated_mean": isolated_mean,
            "mode_isolated_sum": float(np.sum(isolated)) if isolated else float("nan"),
            "mode_isolated_tokens_used": len(isolated),
            "mode_diff_mean": insitu_mean - isolated_mean,
            "log_base": config.log_base,
            "exclude_special": config.exclude_special,
        })

    records_path = config.surprisal_dir / "surprisal_records.jsonl"
    write_jsonl(output_rows, records_path)

    frame = pd.DataFrame(output_rows)
    case_path = config.surprisal_dir / "case_aggregate.csv"
    if not frame.empty:
        summary = (
            frame.groupby("case")
            .agg(
                N_insitu=("mode_insitu_mean", "count"),
                mean_surprisal_insitu=("mode_insitu_mean", "mean"),
                std_surprisal_insitu=("mode_insitu_mean", "std"),
                N_isolated=("mode_isolated_mean", "count"),
                mean_surprisal_isolated=("mode_isolated_mean", "mean"),
                std_surprisal_isolated=("mode_isolated_mean", "std"),
                percent_insitu_below_isolated=("mode_diff_mean", lambda values: float((values < 0).mean() * 100)),
            )
            .reset_index()
        )
        summary.to_csv(case_path, index=False)

    return {
        "records_jsonl": str(records_path),
        "case_csv": str(case_path),
        "num_records": len(output_rows),
    }
