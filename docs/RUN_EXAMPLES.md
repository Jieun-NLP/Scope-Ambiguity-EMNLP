# RUN_EXAMPLES.md

# Running Experiments

This document provides example commands for reproducing all experiments in the paper.

> **Prerequisites**
>
> - Python 3.10+
> - Install all dependencies:
>
> ```bash
> pip install -r requirements.txt
> ```
>
> - Export your Hugging Face access token before running any script.
>
> Linux / macOS
>
> ```bash
> export HF_TOKEN="your_huggingface_token"
> ```
>
> Windows PowerShell
>
> ```powershell
> $env:HF_TOKEN="your_huggingface_token"
> ```

---

# Experiment 1
## Context Effect on Sentence Surprisal

```bash
python scripts/Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Qwen2.5_exp2_sentence.jsonl \
    --output-root results/Qwen2.5-7B \
    --model Qwen/Qwen2.5-7B
```

Skip the spaCy token-level analysis:

```bash
python scripts/Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Qwen2.5_exp2_sentence.jsonl \
    --output-root results/Qwen2.5-7B \
    --model Qwen/Qwen2.5-7B \
    --skip-token-analysis
```

---

# Experiment 2A
## Step 1. Extract Target Sentence Representations

LLaMA 3

```bash
python scripts/extract_sentence_representation_for_Exp2A.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.json \
    --output-file results/Llama3-8B/representations/sentence_means.npz \
    --model meta-llama/Meta-Llama-3-8B \
    --model-type causal_lm \
    --device cuda
```

Mistral

```bash
python scripts/extract_sentence_representation_for_Exp2A.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Mistral_exp2_sentence.jsonl \
    --output-file results/Mistral-7B/representations/sentence_means.npz \
    --model mistralai/Mistral-7B-v0.3 \
    --model-type causal_lm \
    --device cuda
```

Qwen

```bash
python scripts/extract_sentence_representation_for_Exp2A.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Qwen2.5_exp2_sentence.jsonl \
    --output-file results/Qwen2.5-7B/representations/sentence_means.npz \
    --model Qwen/Qwen2.5-7B \
    --model-type causal_lm \
    --device cuda
```

---

# Experiment 2A
## Step 2. Compute Representation Similarity

LLaMA

```bash
python scripts/Experiment2A.py \
    --input-vectors results/Llama3-8B/representations/sentence_means.npz \
    --output-dir results/Llama3-8B/similarity
```

Mistral

```bash
python scripts/Experiment2A.py \
    --input-vectors results/Mistral-7B/representations/sentence_means.npz \
    --output-dir results/Mistral-7B/similarity
```

Qwen

```bash
python scripts/Experiment2A.py \
    --input-vectors results/Qwen2.5-7B/representations/sentence_means.npz \
    --output-dir results/Qwen2.5-7B/similarity
```

Compute metrics only (without plotting):

```bash
python scripts/Experiment2A.py \
    --input-vectors results/Llama3-8B/representations/sentence_means.npz \
    --output-dir results/Llama3-8B/similarity \
    --skip-plot
```

---

# Experiment 2B
## Activation Patching

LLaMA

```bash
python scripts/Experiment2B.py \
    --model meta-llama/Meta-Llama-3-8B \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.json \
    --output-dir results/Llama3-8B/patching
```

Mistral

```bash
python scripts/Experiment2B.py \
    --model mistralai/Mistral-7B-v0.3 \
    --input-data data/processed/processed_data_6set_combined_with_span_Mistral_exp2_sentence.jsonl \
    --output-dir results/Mistral-7B/patching
```

Qwen

```bash
python scripts/Experiment2B.py \
    --model Qwen/Qwen2.5-7B \
    --input-data data/processed/processed_data_6set_combined_with_span_Qwen2.5_exp2_sentence.jsonl \
    --output-dir results/Qwen2.5-7B/patching
```

Run only selected layers:

```bash
python scripts/Experiment2B.py \
    --model meta-llama/Meta-Llama-3-8B \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.json \
    --output-dir results/test \
    --layers 0 8 16 24 31
```

Run only a few sentence pairs for debugging:

```bash
python scripts/Experiment2B.py \
    --model meta-llama/Meta-Llama-3-8B \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.json \
    --output-dir results/test \
    --n-samples 5
```

Skip span repair:

```bash
--skip-span-repair
```

Skip span-mean replacement baseline:

```bash
--skip-span-mean-baseline
```

Skip plotting:

```bash
--skip-plot
```

---

# Experiment 2C

(Coming soon)

```bash
python scripts/Experiment2C.py ...
```

---

# Output Structure

Typical output directories:

```
results/

    Llama3-8B/

        surprisal/

        representations/

        similarity/

        patching/

    Mistral-7B/

        ...

    Qwen2.5-7B/

        ...
```

---

# Recommended Execution Order

```
Experiment 1

↓

Experiment 2A

    Step 1
    extract_sentence_representation_for_Exp2A.py

↓

sentence_means.npz

↓

Experiment2A.py

↓

Experiment 2B

↓

Experiment 2C
```