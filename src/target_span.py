"""Utilities for locating target noun-phrase spans in Experiment 2C.

This module separates *target identification* from representation extraction.
It is adapted from the original Experiment 2C notebook used to locate the
second scopal item (QP2) in SCOPEX.  Character spans are converted to
end-exclusive token spans so they can be passed directly to the generic
representation utilities in :mod:`src.extract_representation`.

The matching procedure intentionally preserves the original experiment's
progressive fallback strategy, including limited determiner/demonstrative
variation for SCOPEX cases 2 and 3.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


# Determiner patterns used by the original QP2 matching notebook.
DET_PATTERNS = [
    r"a", r"an", r"the", r"this", r"that", r"these", r"those",
    r"every", r"most", r"some", r"few", r"several", r"all",
    r"a\s+few", r"the\s+few", r"these\s+few",
    r"all\s+(?:of\s+)?the",
]
DET_A_PATTERNS = [r"a", r"an", r"some", r"a\s+few"]
DET_DEMO_PATTERNS = [r"this", r"these", r"these\s+few"]


def _normalize_quotes(text: str) -> str:
    """Normalize common Unicode quotation marks to ASCII equivalents."""
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _compress_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _strip_zero_width(text: str) -> str:
    """Remove Unicode format characters (category ``Cf``)."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def normalize_for_compare(
    text: str | None,
    *,
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
    normalize_quotes: bool = True,
) -> str:
    """Normalize text for decoded-span validation."""
    normalized = "" if text is None else text
    if normalize_quotes:
        normalized = _normalize_quotes(normalized)
    if normalize_whitespace:
        normalized = _compress_whitespace(normalized)
    if not case_sensitive:
        normalized = normalized.lower()
    return normalized


def load_target_np_lexicon(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the QP2 target-NP lexicon indexed by item ID.

    The expected input is a top-level JSON object whose values are mappings
    containing at least a ``span_text`` field.  Keys are normalized to strings.
    Non-mapping entries are ignored, matching the behavior of the original
    notebook.
    """
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Target-NP lexicon not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise TypeError(
            "The target-NP lexicon must be a top-level JSON object; "
            f"found {type(data).__name__}."
        )

    lexicon: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            lexicon[str(key)] = value

    if not lexicon:
        raise ValueError("The target-NP lexicon is empty after validation.")
    return lexicon


# Backward-compatible name used in the original notebook.
load_qp2_lexicon = load_target_np_lexicon


def span_text_to_flexible_pattern(
    span_text: str,
    *,
    allow_demo_swap: bool = False,
    allow_extra_adjectives: bool = True,
) -> str:
    """Build the flexible NP-matching regex used in the original experiment.

    The fallback pattern allows determiner variation, limited singular/plural
    variation, optional ``types of`` material, and selected adjective changes.
    Demonstrative substitution is enabled only when requested (cases 2 and 3
    in the original SCOPEX extraction).
    """
    tokens = span_text.strip().split()
    determiner = ""
    replace_determiner = False

    det_three = " ".join(tokens[:3]).lower() if len(tokens) >= 3 else ""
    det_two = " ".join(tokens[:2]).lower() if len(tokens) >= 2 else ""
    det_one = tokens[0].lower() if tokens else ""

    if det_three == "all of the":
        determiner = det_three
        tokens = tokens[3:]
        replace_determiner = True
    elif det_two in {"a few", "the few", "these few", "all the"}:
        determiner = det_two
        tokens = tokens[2:]
        replace_determiner = True
    elif det_one in {
        "a", "an", "the", "this", "that", "these", "those", "every",
        "most", "some", "few", "several", "all",
    }:
        determiner = det_one
        tokens = tokens[1:]
        replace_determiner = True

    if tokens and tokens[0].lower() in {"certain", "unique", "particular", "specific"}:
        tokens = tokens[1:]

    pattern = ""
    if replace_determiner:
        if allow_demo_swap and determiner in {"a", "an", "some", "a few"}:
            det_options = DET_A_PATTERNS + DET_DEMO_PATTERNS
        else:
            det_options = DET_PATTERNS
        pattern += r"(?:" + "|".join(det_options) + r")\s+"

    if allow_extra_adjectives:
        pattern += r"(?:(?:certain|unique|particular|specific)\s+)?"

    if (
        len(tokens) >= 2
        and tokens[0].lower() in {"kinds", "types", "sort", "sorts"}
        and tokens[1].lower() == "of"
    ):
        tokens = tokens[2:]

    for index, token in enumerate(tokens):
        token_pattern = re.escape(token)
        if index == len(tokens) - 1 and not token_pattern.endswith(r"\."):
            token_pattern += r"s?"
        pattern += token_pattern

        if index != len(tokens) - 1:
            if allow_extra_adjectives:
                pattern += r"\s+(?:\w+\s+)?"
            else:
                pattern += r"\s+"

    if tokens:
        final_token = re.escape(tokens[-1])
        final_pattern = final_token + r"s?"
        if pattern.endswith(final_pattern):
            prefix = pattern[:-len(final_pattern)]
            with_types = prefix + r"(?:types\s+of\s+)?" + final_pattern
            pattern = r"(?:" + pattern + "|" + with_types + ")"

    return pattern


def find_target_np_char_span(
    full_text: str,
    target_np: str,
    *,
    case_sensitive: bool = True,
    normalize_whitespace: bool = False,
    normalize_quotes: bool = True,
    occurrence: int = -1,
    try_nfkc: bool = True,
    strip_zero_width: bool = True,
    case_number: int | None = None,
    demo_swap_cases: Sequence[int] = (2, 3),
) -> tuple[int, int]:
    """Locate a target NP in ``full_text`` using progressive fallbacks.

    The returned character span follows Python's end-exclusive convention
    ``[start:end)``.  The fallback sequence mirrors the original notebook:
    exact matching, case-insensitive matching, whitespace/quote normalization,
    optional Unicode normalization, and finally flexible NP matching.
    """

    def exact_matches(source: str, target: str, sensitive: bool):
        haystack = source if sensitive else source.lower()
        needle = target if sensitive else target.lower()
        matches: list[tuple[int, int]] = []
        start = 0
        while True:
            position = haystack.find(needle, start)
            if position == -1:
                break
            matches.append((position, position + len(needle)))
            start = position + 1
        return matches

    def whitespace_matches(source: str, target: str, sensitive: bool):
        target_normalized = _compress_whitespace(target)
        pattern = re.escape(target_normalized)
        pattern = re.sub(r"\\\ ", r"\\s+", pattern)
        flags = 0 if sensitive else re.IGNORECASE
        return [(m.start(), m.end()) for m in re.finditer(pattern, source, flags)]

    def pick(matches: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
        if not matches:
            return None
        index = occurrence if occurrence >= 0 else len(matches) - 1
        return matches[index] if 0 <= index < len(matches) else None

    source_original, target_original = full_text, target_np
    source = _normalize_quotes(source_original) if normalize_quotes else source_original
    target = _normalize_quotes(target_original) if normalize_quotes else target_original

    matches = (
        whitespace_matches(source, target, case_sensitive)
        if normalize_whitespace
        else exact_matches(source, target, case_sensitive)
    )
    selected = pick(matches)
    if selected:
        return selected

    matches = (
        whitespace_matches(source, target, False)
        if normalize_whitespace
        else exact_matches(source, target, False)
    )
    selected = pick(matches)
    if selected:
        return selected

    source_quotes = _normalize_quotes(source_original)
    target_quotes = _normalize_quotes(target_original)
    selected = pick(whitespace_matches(source_quotes, target_quotes, False))
    if selected:
        return selected

    source_normalized = source_quotes
    target_normalized = target_quotes
    if strip_zero_width:
        source_normalized = _strip_zero_width(source_normalized)
        target_normalized = _strip_zero_width(target_normalized)
        selected = pick(whitespace_matches(source_normalized, target_normalized, False))
        if selected:
            return selected

    if try_nfkc:
        source_normalized = _nfkc(source_normalized)
        target_normalized = _nfkc(target_normalized)
        selected = pick(whitespace_matches(source_normalized, target_normalized, False))
        if selected:
            return selected

    allow_demo_swap = case_number in set(demo_swap_cases) if case_number is not None else False
    flexible_pattern = span_text_to_flexible_pattern(
        target_original,
        allow_demo_swap=allow_demo_swap,
    )
    matches = [
        (match.start(), match.end())
        for match in re.finditer(flexible_pattern, source_original, flags=re.IGNORECASE)
    ]
    selected = pick(matches)
    if selected:
        return selected

    raise ValueError(
        f"Target NP {target_np!r} was not found in the input after robust matching."
    )


# Backward-compatible name used in the original notebook.
find_span_char_in_full_text_robust = find_target_np_char_span


def tokenize_with_offsets(full_text: str, tokenizer, *, add_special_tokens: bool = True):
    """Tokenize once and return the encoding plus character offsets."""
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    return encoding, encoding["offset_mapping"]


def char_span_to_token_span(
    full_text: str,
    char_span: tuple[int, int],
    offsets: Sequence[tuple[int, int]],
    *,
    mode: str = "snap_out",
    trim_whitespace: bool = True,
) -> tuple[list[int], tuple[int, int], tuple[int, int] | None]:
    """Convert an end-exclusive character span to token coordinates.

    Returns ``(token_indices, snapped_char_span, token_span)``.  ``token_span``
    is end-exclusive and can therefore be used directly for tensor slicing.

    ``mode='snap_out'`` preserves the setting used in the original Experiment
    2C notebook.  The trailing-whitespace condition contains a small typo in
    that notebook; it is corrected here without changing the intended logic.
    """
    char_start, char_end = char_span

    if trim_whitespace:
        while char_start < char_end and full_text[char_start].isspace():
            char_start += 1
        while char_end > char_start and full_text[char_end - 1].isspace():
            char_end -= 1

    token_indices: list[int] = []
    for index, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:  # special token
            continue
        if mode in {"overlap", "snap_out"}:
            if not (end <= char_start or start >= char_end):
                token_indices.append(index)
        elif mode == "inside":
            if char_start <= start and end <= char_end:
                token_indices.append(index)
        else:
            raise ValueError("mode must be one of {'overlap', 'inside', 'snap_out'}")

    snapped_char_span = (char_start, char_end)
    if mode == "snap_out" and token_indices:
        snapped_char_span = (
            min(offsets[index][0] for index in token_indices),
            max(offsets[index][1] for index in token_indices),
        )

    token_span = None
    if token_indices:
        token_span = (min(token_indices), max(token_indices) + 1)

    return token_indices, snapped_char_span, token_span


# Backward-compatible name used in the original notebook.
char_span_to_token_span_from_offs = char_span_to_token_span


def validate_token_span(
    *,
    encoding: Mapping[str, Any],
    tokenizer,
    target_np: str,
    token_span: tuple[int, int],
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
    normalize_quotes: bool = True,
) -> dict[str, Any]:
    """Decode a token span and compare it with the requested target NP."""
    start, end = token_span
    input_ids = encoding["input_ids"]
    if not (0 <= start <= end <= len(input_ids)):
        raise ValueError(
            f"token_span out of range: {(start, end)} for sequence length {len(input_ids)}"
        )

    selected_ids = [int(token_id) for token_id in input_ids[start:end]]
    decoded = tokenizer.decode(
        selected_ids,
        clean_up_tokenization_spaces=False,
        skip_special_tokens=True,
    )

    target_normalized = normalize_for_compare(
        target_np,
        case_sensitive=case_sensitive,
        normalize_whitespace=normalize_whitespace,
        normalize_quotes=normalize_quotes,
    )
    decoded_normalized = normalize_for_compare(
        decoded,
        case_sensitive=case_sensitive,
        normalize_whitespace=normalize_whitespace,
        normalize_quotes=normalize_quotes,
    )

    return {
        "token_span": (start, end),
        "token_ids": selected_ids,
        "decoded_text": decoded,
        "decoded_well": target_normalized == decoded_normalized,
        "target_normalized": target_normalized,
        "decoded_normalized": decoded_normalized,
    }


def get_target_np_token_span(
    *,
    full_text: str,
    target_np: str,
    tokenizer,
    case_number: int | None = None,
    occurrence: int = -1,
    mode: str = "snap_out",
    add_special_tokens: bool = True,
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
    normalize_quotes: bool = True,
    try_nfkc: bool = True,
    strip_zero_width: bool = True,
) -> dict[str, Any]:
    """Locate one target NP and return all span information needed by Exp. 2C.

    This is the main public API for target-span extraction.  It combines the
    original notebook's robust character matching, offset-based token mapping,
    and decoded-span validation.
    """
    char_span = find_target_np_char_span(
        full_text,
        target_np,
        case_sensitive=case_sensitive,
        normalize_whitespace=normalize_whitespace,
        normalize_quotes=normalize_quotes,
        occurrence=occurrence,
        try_nfkc=try_nfkc,
        strip_zero_width=strip_zero_width,
        case_number=case_number,
        demo_swap_cases=(2, 3),
    )

    encoding, offsets = tokenize_with_offsets(
        full_text,
        tokenizer,
        add_special_tokens=add_special_tokens,
    )
    token_indices, snapped_char_span, token_span = char_span_to_token_span(
        full_text,
        char_span,
        offsets,
        mode=mode,
        trim_whitespace=True,
    )
    if token_span is None:
        raise ValueError(
            f"No tokens overlap target NP {target_np!r} at character span {char_span}."
        )

    validation = validate_token_span(
        encoding=encoding,
        tokenizer=tokenizer,
        target_np=target_np,
        token_span=token_span,
        case_sensitive=case_sensitive,
        normalize_whitespace=normalize_whitespace,
        normalize_quotes=normalize_quotes,
    )

    return {
        "target_np": target_np,
        "char_span": char_span,
        "snapped_char_span": snapped_char_span,
        "token_indices": token_indices,
        "token_span": token_span,
        "decoded_text": validation["decoded_text"],
        "decoded_well": validation["decoded_well"],
    }


def get_scopex_target_np_token_span(
    record: Mapping[str, Any],
    tokenizer,
    target_np_lexicon: Mapping[str, Mapping[str, Any]],
    *,
    id_key: str = "id",
    case_key: str = "case",
    text_key: str = "full_text",
    span_text_key: str = "span_text",
    **span_kwargs,
) -> dict[str, Any]:
    """Resolve SCOPEX's QP2 target NP from a record and return its token span.

    This thin wrapper keeps the SCOPEX-specific lexicon lookup outside the
    generic representation-extraction code.
    """
    item_id = str(record.get(id_key, record.get("idx")))
    if item_id not in target_np_lexicon:
        raise KeyError(f"Item ID {item_id!r} is not present in the target-NP lexicon.")

    lexicon_entry = target_np_lexicon[item_id]
    target_np = lexicon_entry.get(span_text_key)
    if not isinstance(target_np, str) or not target_np.strip():
        raise ValueError(f"Missing target NP for item ID {item_id!r}.")

    full_text = record.get(text_key)
    if not isinstance(full_text, str) or not full_text:
        raise ValueError(f"Missing {text_key!r} for item ID {item_id!r}.")

    case_value = record.get(case_key)
    case_number = int(case_value) if case_value is not None else None

    result = get_target_np_token_span(
        full_text=full_text,
        target_np=target_np,
        tokenizer=tokenizer,
        case_number=case_number,
        **span_kwargs,
    )
    result.update({"id": item_id, "case": case_value})
    return result
