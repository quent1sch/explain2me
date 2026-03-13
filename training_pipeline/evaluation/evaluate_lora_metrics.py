"""
evaluate_lora_metrics.py

Evaluation script for the Explain2Me LoRA adapter.

PURPOSE
This script evaluates a LoRA-adapted instruction-tuned LLaMA model designed to
generate explanations tailored to different audiences (e.g., child, student,
domain expert). It compares the base model and the LoRA model on structural
alignment, semantic fidelity, and audience-targeted readability.


METRICS

1. ROUGE-L
   Measures overlap between generated outputs and reference explanations.
   Captures structural similarity of the explanation.

2. BERTScore (F1)
   Measures semantic similarity using contextual embeddings.
   Verifies that meaning is preserved even when phrasing changes.

3. Readability Errors
   All readability metrics (Flesch-Kincaid, Dale-Chall, Gunning Fog, SMOG, ARI) 
   are expressed as errors relative to the reference text:

       error = predicted_grade - reference_grade

   This reflects how much the generated explanation deviates in complexity from the
   target text. Positive values indicate outputs that are too complex; negative values 
   indicate oversimplification.

4. Composite Readability Error
   Defined as the mean of all individual readability errors.
   Provides a robust single measure capturing multiple aspects of text difficulty.

For each metric, we compute:
- base_mean / lora_mean: mean error across all samples
- base_std / lora_std: standard deviation across all samples
- delta: difference between LoRA and base mean

These statistics give a clear picture of both accuracy and consistency of the model's 
audience-targeted explanations.

OUTPUTS
- eval_lora_per_sample.json (metrics for each example)
- eval_lora_summary.json (mean and std statistics for base vs LoRA)
"""

import os
import json
import numpy as np
from rouge_score import rouge_scorer
import bert_score
import textstat
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------

EVAL_OUTPUT_DIR = "training_pipeline/evaluation/evaluation_results"
GENERATED_OUTPUT_PATH = "training_pipeline/evaluation/evaluation_results/generate_outputs.json"

PER_SAMPLE_FILE = os.path.join(EVAL_OUTPUT_DIR, "eval_lora_per_sample.json")
SUMMARY_FILE = os.path.join(EVAL_OUTPUT_DIR, "eval_lora_summary.json")

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------

# retrieve generated outputs from base & lora models - To be used for evaluation
with open(GENERATED_OUTPUT_PATH, "r") as f:
    eval_inputs = json.load(f)

# ---------------------------
# ROUGE-L (SCORER) (n-gram similarity metric)
# ---------------------------

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
# Porter stemmer used to strip word suffixes to improve matching

# ---------------------------
# READABILITY METRICS
# ---------------------------

READABILITY_METRICS = {
    "fk_grade": textstat.flesch_kincaid_grade,
    "dale_chall": textstat.dale_chall_readability_score,
    "gunning_fog": textstat.gunning_fog,
    "smog": textstat.smog_index,
    "ari": textstat.automated_readability_index
}

# ---------------------------
# PREPARE TEXT LISTS
# ---------------------------

references = [r["reference"] for r in eval_inputs]
base_outputs = [r["base_output"] for r in eval_inputs]
lora_outputs = [r["lora_output"] for r in eval_inputs]

# ---------------------------
# COMPUTE BERTSCORE (semantic similarity) (vectorized)
# => tokenize -> embeddings with BERT family model -> compute cosine similarity ->
#    -> soft matching alignment -> produces precision, recall, f1 score
# ---------------------------

P_base, R_base, F1_base = bert_score.score(
    base_outputs,
    references,
    lang="en",
    verbose=True
)

P_lora, R_lora, F1_lora = bert_score.score(
    lora_outputs,
    references,
    lang="en",
    verbose=True
)

# ---------------------------
# PER-SAMPLE METRICS
# ---------------------------

per_sample_results = []

print("Computing per-sample metrics...")

for i, sample in enumerate(tqdm(eval_inputs)):

    ref = sample["reference"]
    base = sample["base_output"]
    lora = sample["lora_output"]

    # ROUGE
    base_rouge = scorer.score(ref, base)["rougeL"].fmeasure
    lora_rouge = scorer.score(ref, lora)["rougeL"].fmeasure

    # BERTScore
    base_bert = F1_base[i].item()
    lora_bert = F1_lora[i].item()

    # READABILITY SCORES
    ref_scores = {}
    base_scores = {}
    lora_scores = {}

    for name, fn in READABILITY_METRICS.items():
        ref_scores[name] = fn(ref)
        base_scores[name] = fn(base)
        lora_scores[name] = fn(lora)

    # COMPOSITE READABILITY (mean of readability scores per sample)
    ref_composite = np.mean(list(ref_scores.values()))
    base_composite = np.mean(list(base_scores.values()))
    lora_composite = np.mean(list(lora_scores.values()))

    # All these readability measures makes sens individually, but when aggregated, it is 
    # meaningless, thus we use deviation from target grade.
    # Ideally, user stated grade (e.g. mapping from age to grade) would be used as 
    # target grade. But, as user age/characteristics were generated with rather 
    # "lack of care" in "data_pipeline/lora_train_data_formatting.py", we consider
    # more fit to use the metrics for the reference text (target text).

    # measure targeting error
    base_errors = {
        f"{name}_error": base_scores[name] - ref_scores[name]
        for name in READABILITY_METRICS
    }

    lora_errors = {
        f"{name}_error": lora_scores[name] - ref_scores[name]
        for name in READABILITY_METRICS
    }

    # Composite readability error
    base_errors["composite_readability_error"] = base_composite - ref_composite
    lora_errors["composite_readability_error"] = lora_composite - ref_composite

    # RESULT
    result = {
        "id": i,
        "base": {
            "rougeL": base_rouge,
            "bertscore_f1": base_bert,
            **base_errors
        },
        "lora": {
            "rougeL": lora_rouge,
            "bertscore_f1": lora_bert,
            **lora_errors
        }
    }

    per_sample_results.append(result)

# ---------------------------
# SAVE PER-SAMPLE METRICS
# ---------------------------

with open(PER_SAMPLE_FILE, "w") as f:
    json.dump(per_sample_results, f, indent=2)

# ---------------------------
# SUMMARY METRICS
# ---------------------------

def mean_std(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))

def collect(metric, model):
    return [
        r[model][metric]
        for r in per_sample_results
        if metric in r[model] and r[model][metric] is not None
    ]

summary = {}

metrics = [
    "rougeL",
    "bertscore_f1",
    "fk_grade_error",
    "dale_chall_error",
    "gunning_fog_error",
    "smog_error",
    "ari_error",
    "composite_readability_error"
]

for metric in metrics:
    base_vals = collect(metric, "base")
    lora_vals = collect(metric, "lora")

    base_mean, base_std = mean_std(base_vals)
    lora_mean, lora_std = mean_std(lora_vals)

    summary[metric] = {
        "base_mean": base_mean,
        "base_std": base_std,
        "lora_mean": lora_mean,
        "lora_std": lora_std,
        "delta": None if base_mean is None or lora_mean is None else lora_mean - base_mean
    }

# ---------------------------
# SAVE SUMMARY
# ---------------------------

with open(SUMMARY_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary metrics saved to: {SUMMARY_FILE}")

# ---------------------------
# PRINT SUMMARY
# ---------------------------

print("\nEvaluation Summary:\n")
print(json.dumps(summary, indent=2))