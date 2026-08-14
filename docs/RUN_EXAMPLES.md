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

### LLaMA-3-8B

```bash
python scripts/Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --output-root results/Llama3-8B \
    --model meta-llama/Meta-Llama-3-8B
```

### Mistral-7B-v0.3

```bash
python scripts/Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Mistral.jsonl \
    --output-root results/Mistral-7B \
    --model mistralai/Mistral-7B-v0.3
```

### Qwen2.5-7B

```bash
python scripts/Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Qwen2.5.jsonl \
    --output-root results/Qwen2.5-7B \
    --model Qwen/Qwen2.5-7B
```

Skip the spaCy token-level analysis:

```bash
python scripts/Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Qwen2.5.jsonl \
    --output-root results/Qwen2.5-7B \
    --model Qwen/Qwen2.5-7B \
    --skip-token-analysis
```

---

# Experiment 2A

## Step 1. Extract Target Sentence Representations

### LLaMA-3-8B

```bash
python scripts/extract_sentence_representation_for_Exp2A.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --output-file results/Llama3-8B/representations/sentence_means.npz \
    --model meta-llama/Meta-Llama-3-8B \
    --model-type causal_lm \
    --device cuda
```

### Mistral-7B-v0.3

```bash
python scripts/extract_sentence_representation_for_Exp2A.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Mistral.jsonl \
    --output-file results/Mistral-7B/representations/sentence_means.npz \
    --model mistralai/Mistral-7B-v0.3 \
    --model-type causal_lm \
    --device cuda
```

### Qwen2.5-7B

```bash
python scripts/extract_sentence_representation_for_Exp2A.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Qwen2.5.jsonl \
    --output-file results/Qwen2.5-7B/representations/sentence_means.npz \
    --model Qwen/Qwen2.5-7B \
    --model-type causal_lm \
    --device cuda
```

---

## Step 2. Compute Representation Similarity

```bash
python scripts/Experiment2A.py \
    --input-vectors results/Llama3-8B/representations/sentence_means.npz \
    --output-dir results/Llama3-8B/similarity
```

Replace the input and output paths for Mistral and Qwen.

---

# Experiment 2B

## Activation Patching

### LLaMA-3-8B

```bash
python scripts/Experiment2B.py \
    --model meta-llama/Meta-Llama-3-8B \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --output-dir results/Llama3-8B/patching
```

### Mistral-7B-v0.3

```bash
python scripts/Experiment2B.py \
    --model mistralai/Mistral-7B-v0.3 \
    --input-data data/processed/processed_data_6set_combined_with_span_Mistral.jsonl \
    --output-dir results/Mistral-7B/patching
```

### Qwen2.5-7B

```bash
python scripts/Experiment2B.py \
    --model Qwen/Qwen2.5-7B \
    --input-data data/processed/processed_data_6set_combined_with_span_Qwen2.5.jsonl \
    --output-dir results/Qwen2.5-7B/patching
```

---

# Experiment 2C

## Step 0. Prepare Tokenizer-Specific QP2 Spans

The updated QP2 lexicon:

```text
data/processed/extracted_qp2s_updated.json
```

### LLaMA-3-8B

```bash
python scripts/prepare_target_np_spans_for_Exp2C.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --qp2-lexicon data/processed/extracted_qp2s_updated.json \
    --output-data data/processed/processed_data_Exp2C_Llama3.jsonl \
    --model meta-llama/Meta-Llama-3-8B
```

### Mistral-7B-v0.3

```bash
python scripts/prepare_target_np_spans_for_Exp2C.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Mistral.jsonl \
    --qp2-lexicon data/processed/extracted_qp2s_updated.json \
    --output-data data/processed/processed_data_Exp2C_Mistral.jsonl \
    --model mistralai/Mistral-7B-v0.3
```

### Qwen2.5-7B

```bash
python scripts/prepare_target_np_spans_for_Exp2C.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Qwen2.5.jsonl \
    --qp2-lexicon data/processed/extracted_qp2s_updated.json \
    --output-data data/processed/processed_data_Exp2C_Qwen2.5.jsonl \
    --model Qwen/Qwen2.5-7B
```

---

## Step 1. Extract SPECIFICITY Features

### LLaMA-3-8B

```bash
python scripts/extract_specificity_representation_for_Exp2C.py \
    --input-data data/specificity/synthetic_specificity_dataset.jsonl \
    --output-dir outputs/Exp2C/Llama3-8B/specificity_features \
    --model meta-llama/Meta-Llama-3-8B \
    --device cuda
```

Replace the model name and output directory for Mistral and Qwen.

---

## Step 2. Train the Scalar-Mixing Specificity Probe

```bash
python scripts/train_specificity_probe_for_Exp2C.py \
    --feature-dir outputs/Exp2C/Llama3-8B/specificity_features \
    --output-dir outputs/Exp2C/Llama3-8B/specificity_probe \
    --device cuda
```

Generated files:

```text
specificity_edge_probe.pt
validation_results.json
training_history.json
```

---

## Step 3. Run Frozen Inference on SCOPEX

### LLaMA-3-8B

```bash
python scripts/infer_specificity_probe_for_Exp2C.py \
    --scopex-data data/processed/processed_data_Exp2C_Llama3.jsonl \
    --checkpoint outputs/Exp2C/Llama3-8B/specificity_probe/specificity_edge_probe.pt \
    --output-dir outputs/Exp2C/Llama3-8B/scopex_inference \
    --model meta-llama/Meta-Llama-3-8B \
    --device cuda
```

### Mistral-7B-v0.3

```bash
python scripts/infer_specificity_probe_for_Exp2C.py \
    --scopex-data data/processed/processed_data_Exp2C_Mistral.jsonl \
    --checkpoint outputs/Exp2C/Mistral-7B-v0.3/specificity_probe/specificity_edge_probe.pt \
    --output-dir outputs/Exp2C/Mistral-7B-v0.3/scopex_inference \
    --model mistralai/Mistral-7B-v0.3 \
    --device cuda
```

### Qwen2.5-7B

```bash
python scripts/infer_specificity_probe_for_Exp2C.py \
    --scopex-data data/processed/processed_data_Exp2C_Qwen2.5.jsonl \
    --checkpoint outputs/Exp2C/Qwen2.5-7B/specificity_probe/specificity_edge_probe.pt \
    --output-dir outputs/Exp2C/Qwen2.5-7B/scopex_inference \
    --model Qwen/Qwen2.5-7B \
    --device cuda
```

---

# Output Structure

```text
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

```text
Experiment 1
        ↓
Experiment 2A
        ↓
Experiment 2B
        ↓
Experiment 2C

prepare_target_np_spans_for_Exp2C.py
        ↓
extract_specificity_representation_for_Exp2C.py
        ↓
train_specificity_probe_for_Exp2C.py
        ↓
infer_specificity_probe_for_Exp2C.py
```