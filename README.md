# Scope Ambiguity in Large Language Models

Code and data for experiments on context-dependent representations of
scope ambiguity in large language models.

The repository contains the implementation of the SCOPEX experiments
described in the paper, including:

-   **Experiment 1:** surprisal-based contextual disambiguation
-   **Experiment 2A:** layerwise representation similarity
-   **Experiment 2B:** activation patching
-   **Experiment 2C:** scalar-mixing edge probing for scopal specificity

The experiments are implemented for:

-   LLaMA-3-8B
-   Mistral-7B-v0.3
-   Qwen2.5-7B

For complete example commands, see `docs/RUN_EXAMPLES.md`.

------------------------------------------------------------------------

## Environment

Python **3.11+** is recommended.

Install the required packages with:

``` bash
pip install -r requirements.txt
```

Experiment 1 additionally uses the English spaCy pipeline for
token-level linguistic analysis:

``` bash
python -m spacy download en_core_web_sm
```

The spaCy analysis can be skipped with the corresponding command-line
option described in `docs/RUN_EXAMPLES.md`.

------------------------------------------------------------------------

## Hugging Face Authentication

The experiments use Hugging Face models, including gated models such as
LLaMA-3.

Set a Hugging Face access token before running the scripts.

### Linux / macOS

``` bash
export HF_TOKEN="your_huggingface_token"
```

### Windows PowerShell

``` powershell
$env:HF_TOKEN="your_huggingface_token"
```

Make sure that the token has permission to access the model being used.

------------------------------------------------------------------------

## Repository Structure

``` text
Scope-Ambiguity-EMNLP/
├── data/
│   ├── processed/
│   └── specificity/
├── docs/
│   └── RUN_EXAMPLES.md
├── scripts/
│   ├── Experiment1.py
│   ├── extract_sentence_representation_for_Exp2A.py
│   ├── Experiment2A.py
│   ├── Experiment2B.py
│   ├── prepare_target_np_spans_for_Exp2C.py
│   ├── extract_specificity_representation_for_Exp2C.py
│   ├── train_specificity_probe_for_Exp2C.py
│   └── infer_specificity_probe_for_Exp2C.py
├── src/
│   ├── probing.py
│   ├── target_span.py
│   └── ...
├── README.md
└── requirements.txt
```

Reusable functions shared across experiments are placed under `src/`,
while executable experiment pipelines are under `scripts/`.

------------------------------------------------------------------------

## Data

### SCOPEX

The processed SCOPEX datasets used by Experiments 1, 2A, and 2B are:

``` text
data/processed/processed_data_6set_combined_with_span_Llama3.jsonl
data/processed/processed_data_6set_combined_with_span_Mistral.jsonl
data/processed/processed_data_6set_combined_with_span_Qwen2.5.jsonl
```

Each model has its own processed file because token indices depend on
the tokenizer.

Experiment 2C uses target-NP-specific versions of SCOPEX:

``` text
data/processed/processed_data_Exp2C_Llama3.jsonl
data/processed/processed_data_Exp2C_Mistral.jsonl
data/processed/processed_data_Exp2C_Qwen2.5.jsonl
```

The QP2 lexicon used for Experiment 2C preprocessing is:

``` text
data/processed/extracted_qp2s_updated.json
```

### SPECIFICITY

The independent dataset used to train the Experiment 2C probe is:

``` text
data/specificity/synthetic_specificity_dataset.jsonl
```

The specificity probe is trained and validated only on this dataset
before being applied to SCOPEX.

------------------------------------------------------------------------

## Experiment 1: Surprisal-Based Disambiguation

Experiment 1 measures how preceding contexts affect sentence-level and
token-level surprisal across the six SCOPEX conditions.

Example:

``` bash
python scripts/Experiment1.py \
    --input data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --output-root results/Llama3-8B \
    --model meta-llama/Meta-Llama-3-8B
```

Generated files include:

``` text
results/Llama3-8B/surprisal/
├── token_deltas_insitu_spacy_all.jsonl
├── surprisal_records.jsonl
└── case_aggregate.csv
```

The generated outputs can be used to reproduce the sentence-level
statistics and scope-sensitivity analyses reported in the paper.

------------------------------------------------------------------------

## Experiment 2A: Layerwise Representation Similarity

Experiment 2A compares hidden representations across SCOPEX conditions
using layerwise similarity measures.

The experiment consists of two stages:

1.  Extract target-sentence representations.
2.  Compute representation similarity.

Example representation extraction:

``` bash
python scripts/extract_sentence_representation_for_Exp2A.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --output-file results/Llama3-8B/representations/sentence_means.npz \
    --model meta-llama/Meta-Llama-3-8B \
    --model-type causal_lm \
    --device cuda
```

Similarity is then computed with `scripts/Experiment2A.py`.

The implementation computes layerwise representation similarity using
cosine similarity and centered kernel alignment (CKA).

The resulting figures reproduce the representation-similarity analyses
reported in the paper.

------------------------------------------------------------------------

## Experiment 2B: Activation Patching

Experiment 2B tests whether context-conditioned sentence representations
have a causal effect on downstream prediction.

The implementation follows four methodological principles:

1.  Precomputed token IDs generated during preprocessing are used
    directly.
2.  Sentence spans are **not reconstructed** during the patching stage.
3.  Position-wise patching is performed only when the stored
    target-sentence token IDs are identical across conditions.
4.  Alignment mismatches are reported explicitly.

Example:

``` bash
python scripts/Experiment2B.py \
    --model meta-llama/Meta-Llama-3-8B \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --output-dir results/Llama3-8B/patching
```

Generated files:

``` text
results/Llama3-8B/patching/
├── patching_results.csv
├── layer_stats.csv
├── span_mean_replacement_results.csv
├── excluded_alignment_mismatches.json
└── activation_patching_results.png
```

The implementation supports:

-   Bidirectional activation patching
-   Selected-layer runs
-   Debugging subsets
-   The span-mean replacement baseline

Unlike earlier implementations that reconstructed sentence spans at
runtime, the current implementation uses the token IDs and sentence
spans stored during preprocessing.

This design ensures that Experiments 1, 2A, and 2B operate on exactly
the same tokenization and sentence-boundary definitions.

------------------------------------------------------------------------

## Experiment 2C: Probing for Scopal Specificity

Experiment 2C tests whether specificity is recoverable from the hidden
representations of the target NP, defined as the second scopal item
(QP2).

The main analysis uses a **scalar-mixing edge probe**. For each model,
target-NP representations are mean-pooled within each layer, combined
through learned softmax-normalized scalar-mixing weights, and passed to
a two-layer MLP binary classifier.

The probe is trained on the independent SPECIFICITY dataset. The
validation split is used to select the decision threshold. The trained
probe and threshold are then frozen and applied to SCOPEX without
further optimization.

Experiment 2C consists of four stages.

### Step 0. Prepare Tokenizer-Specific Target-NP Spans

``` bash
python scripts/prepare_target_np_spans_for_Exp2C.py \
    --input-data data/processed/processed_data_6set_combined_with_span_Llama3.jsonl \
    --qp2-lexicon data/processed/extracted_qp2s_updated.json \
    --output-data data/processed/processed_data_Exp2C_Llama3.jsonl \
    --model meta-llama/Meta-Llama-3-8B
```

### Step 1. Extract SPECIFICITY Representations

``` bash
python scripts/extract_specificity_representation_for_Exp2C.py \
    --input-data data/specificity/synthetic_specificity_dataset.jsonl \
    --output-dir outputs/Exp2C/Llama3-8B/specificity_features \
    --model meta-llama/Meta-Llama-3-8B \
    --device cuda
```

### Step 2. Train the Scalar-Mixing Probe

``` bash
python scripts/train_specificity_probe_for_Exp2C.py \
    --feature-dir outputs/Exp2C/Llama3-8B/specificity_features \
    --output-dir outputs/Exp2C/Llama3-8B/specificity_probe \
    --device cuda
```

### Step 3. Frozen Inference on SCOPEX

``` bash
python scripts/infer_specificity_probe_for_Exp2C.py \
    --scopex-data data/processed/processed_data_Exp2C_Llama3.jsonl \
    --checkpoint outputs/Exp2C/Llama3-8B/specificity_probe/specificity_edge_probe.pt \
    --output-dir outputs/Exp2C/Llama3-8B/scopex_inference \
    --model meta-llama/Meta-Llama-3-8B \
    --device cuda
```

The same pipeline is run independently for Mistral-7B-v0.3 and
Qwen2.5-7B.

------------------------------------------------------------------------

## Reproducing the Experiments

``` text
Experiment 1

Experiment 2A
    extract_sentence_representation_for_Exp2A.py
    -> Experiment2A.py

Experiment 2B
    Experiment2B.py

Experiment 2C
    prepare_target_np_spans_for_Exp2C.py
    -> extract_specificity_representation_for_Exp2C.py
    -> train_specificity_probe_for_Exp2C.py
    -> infer_specificity_probe_for_Exp2C.py
```

See `docs/RUN_EXAMPLES.md` for model-specific commands and optional
arguments.

------------------------------------------------------------------------

## Generated Outputs

Experiment outputs are written to directories such as:

``` text
results/
outputs/
```

These directories contain derived artifacts such as hidden
representations, trained probe checkpoints, intermediate summaries, and
inference results.

------------------------------------------------------------------------

## Notes on Reproducibility

-   Use the model-specific processed SCOPEX file matching the
    tokenizer/model being evaluated.
-   Experiments 1, 2A, and 2B use the tokenizer-specific token IDs and
    sentence spans generated during preprocessing.
-   Experiment 2C trains a separate specificity probe for each language
    model.
-   The Experiment 2C probe is trained only on SPECIFICITY; SCOPEX is
    used only for frozen inference.
-   Tokenizer-specific target-NP spans should be prepared before running
    Experiment 2C inference.
-   For exact command-line examples, refer to `docs/RUN_EXAMPLES.md`.
