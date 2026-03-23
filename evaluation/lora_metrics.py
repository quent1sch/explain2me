"""
lora_metrics.py

Evaluation module for the Explain2Me LoRA adapter.

PURPOSE
This class-based module evaluates a LoRA-adapted instruction-tuned LLaMA model 
designed to generate explanations for different audiences (e.g., child, student, 
domain expert). It compares outputs from the base model and the LoRA-adapted model 
in terms of structural alignment, semantic fidelity, and readability relative to 
the reference explanations.

METRICS
- ROUGE-L: measures overlap between generated outputs and references.
- BERTScore (F1): measures semantic similarity with contextual embeddings.
- Readability Errors: compares readability scores (Flesch-Kincaid, Dale-Chall, Gunning Fog, SMOG, ARI) 
  against the reference text. Composite readability error is computed as the mean deviation.

For each metric, the module computes:
- base_mean / lora_mean: mean score across all samples
- base_std / lora_std: standard deviation
- delta: difference between LoRA and base mean

OUTPUTS
- eval_lora_per_sample.json: per-sample metrics
- eval_lora_summary.json: aggregated mean/std statistics for base vs LoRA

PORTABILITY
- Uses pathlib for cross-platform path handling
- Supports relative paths and .env configuration for directories
- Works in Docker, cloud VMs, and Colab without modification

USAGE
from evaluation.lora_metrics import LoRAMetrics
metrics = LoRAMetrics(generated_output_path="data/generated_outputs.json",
                      eval_output_dir="evaluation/evaluation_results")
summary = metrics.run()
"""



from pathlib import Path
import json
import numpy as np
from rouge_score import rouge_scorer
import bert_score
import textstat
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

READABILITY_METRICS = {
    "fk_grade": textstat.flesch_kincaid_grade,
    "dale_chall": textstat.dale_chall_readability_score,
    "gunning_fog": textstat.gunning_fog,
    "smog": textstat.smog_index,
    "ari": textstat.automated_readability_index
}

class LoRAMetrics:
    """Self-contained LoRA evaluation: computes metrics, saves per-sample + summary JSON."""

    def __init__(self, generated_output_path=None, eval_output_dir=None):
        base_dir = Path(__file__).resolve().parent

        # Load paths from .env if not provided
        self.generated_output_path = Path(
            generated_output_path or os.getenv("GENERATED_OUTPUT_PATH", "generated_outputs.json")
        )
        self.eval_output_dir = Path(
            eval_output_dir or os.getenv("EVAL_OUTPUT_DIR", "evaluation_results")
        )

        # Resolve relative paths
        if not self.generated_output_path.is_absolute():
            self.generated_output_path = base_dir / self.generated_output_path

        if not self.eval_output_dir.is_absolute():
            self.eval_output_dir = base_dir / self.eval_output_dir

        # Ensure output directory exists
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        self.per_sample_file = self.eval_output_dir / "eval_lora_per_sample.json"
        self.summary_file = self.eval_output_dir / "eval_lora_summary.json"

    def _load_data(self):
        # retrieve generated outputs from base & lora models - To be used for evaluation
        with self.generated_output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # prepare text lists
        references = [r["reference"] for r in data]
        base_outputs = [r["base_output"] for r in data]
        lora_outputs = [r["lora_output"] for r in data]
        return data, references, base_outputs, lora_outputs

    def _compute_bertscore(self, base_outputs, lora_outputs, references):
        # COMPUTE BERTSCORE (semantic similarity) (vectorized)
        # => tokenize -> embeddings with BERT family model -> compute cosine similarity ->
        #    -> soft matching alignment -> produces precision, recall, f1 score
        P_base, R_base, F1_base = bert_score.score(
            base_outputs, references, lang="en", verbose=True
        )
        P_lora, R_lora, F1_lora = bert_score.score(
            lora_outputs, references, lang="en", verbose=True
        )
        return F1_base, F1_lora

    def _compute_per_sample(self, data, references, base_outputs, lora_outputs):
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        F1_base, F1_lora = self._compute_bertscore(base_outputs, lora_outputs, references)
        per_sample_results = []

        for i, ref in enumerate(tqdm(references, desc="Computing per-sample metrics")):
            base = base_outputs[i]
            lora = lora_outputs[i]

            # ROUGE-L
            base_rouge = scorer.score(ref, base)["rougeL"].fmeasure
            lora_rouge = scorer.score(ref, lora)["rougeL"].fmeasure

            # BERTScore
            base_bert = F1_base[i].item()
            lora_bert = F1_lora[i].item()

            # Readability
            ref_scores = {name: fn(ref) for name, fn in READABILITY_METRICS.items()}
            base_scores = {name: fn(base) for name, fn in READABILITY_METRICS.items()}
            lora_scores = {name: fn(lora) for name, fn in READABILITY_METRICS.items()}

            # Composite readability
            ref_comp = np.mean(list(ref_scores.values()))
            base_comp = np.mean(list(base_scores.values()))
            lora_comp = np.mean(list(lora_scores.values()))

            # All these readability measures makes sens individually, but when aggregated, it is 
            # meaningless, thus we use deviation from target grade.
            # Ideally, user stated grade (e.g. mapping from age to grade) would be used as 
            # target grade. But, as user age/characteristics were generated with rather 
            # "lack of care" in "data_pipeline/lora_train_data_formatting.py", we consider
            # more fit to use the metrics for the reference text (target text).

            # measure targeting error
            base_errors = {f"{k}_error": base_scores[k] - ref_scores[k] for k in READABILITY_METRICS}
            lora_errors = {f"{k}_error": lora_scores[k] - ref_scores[k] for k in READABILITY_METRICS}
            
            # Composite readability error
            base_errors["composite_readability_error"] = base_comp - ref_comp
            lora_errors["composite_readability_error"] = lora_comp - ref_comp

            result = {
                "id": i,
                "base": {"rougeL": base_rouge, "bertscore_f1": base_bert, **base_errors},
                "lora": {"rougeL": lora_rouge, "bertscore_f1": lora_bert, **lora_errors}
            }
            per_sample_results.append(result)

        return per_sample_results

    @staticmethod
    def _compute_summary(per_sample_results):
        def mean_std(values):
            vals = [v for v in values if v is not None]
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))

        def collect(metric, model):
            return [r[model][metric] for r in per_sample_results if metric in r[model]]

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

        return summary

    def run(self):
        """Full evaluation: load, compute per-sample metrics, compute summary, save results."""
        data, references, base_outputs, lora_outputs = self._load_data()
        per_sample_results = self._compute_per_sample(data, references, base_outputs, lora_outputs)

        # Save outputs
        with self.per_sample_file.open("w", encoding="utf-8") as f:
            json.dump(per_sample_results, f, indent=2)

        summary = self._compute_summary(per_sample_results)
        with self.summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Per-sample metrics saved to: {self.per_sample_file}")
        print(f"Summary metrics saved to: {self.summary_file}")
        
        return summary