# Scope Ambiguity in Large Language Models

Official implementation of the EMNLP 2026 paper

> Context-dependent Representation of Scope Ambiguity in Large Language Models

## Environment

Python 3.10+

pip install -r requirements.txt

## Hugging Face Authentication

The experiments require access to gated Hugging Face models.

Export your Hugging Face access token before running the scripts.

Linux / macOS

export HF_TOKEN=your_token

Windows PowerShell

$env:HF_TOKEN="your_token"

## Dataset

Place the processed dataset under

data/processed/

Example:

data/processed/
    processed_data_6set_combined_with_span_Qwen2.5_exp2_sentence.jsonl

## Run Experiment 1

python Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Qwen2.5_exp2_sentence.jsonl \
    --output-root results/Qwen2.5-7B \
    --model Qwen/Qwen2.5-7B

To skip spaCy-based token analysis:

python Experiment1.py \
    --input data/processed/... \
    --output-root results/Qwen2.5-7B \
    --model Qwen/Qwen2.5-7B \
    --skip-token-analysis

To enable token-level linguistic analysis:

pip install spacy

python -m spacy download en_core_web_sm

## Output

The script produces

results/

    surprisal_records.jsonl

    case_aggregate.csv

    token_deltas_insitu_spacy_all.jsonl

    ...