"""
main_evaluation.py

Unified entry point for evaluation workflows in the Explain2Me project.

This script orchestrates the full evaluation pipeline, from generating model
outputs to computing metrics and running external evaluations.

MODES
-----
- generate : Generate outputs from base and LoRA models on test data
- lora     : Compute local metrics (ROUGE-L, BERTScore, readability errors)
- judge    : Evaluate outputs using an external LLM (Hugging Face API)
- trainer  : Evaluate model using Trainer (loss, perplexity) with W&B logging

WORKFLOW
--------
Typical evaluation pipeline:

    1. Generate outputs
        → python main_evaluation.py --mode generate

    2. Run local metrics evaluation
        → python main_evaluation.py --mode lora

    3. Run LLM-as-a-judge evaluation
        → python main_evaluation.py --mode judge

    4. (Optional) Run trainer-based evaluation
        → python main_evaluation.py --mode trainer

This modular design allows each step to be run independently, making the
pipeline flexible and easy to debug or extend.

RESULTS
-------
All outputs are stored in the evaluation directory (configured via .env):

- generated_outputs.json        : model generations (base vs LoRA)
- eval_lora_per_sample.json    : per-sample local metrics
- eval_lora_summary.json       : aggregated local metrics
- judge_scores.json            : per-sample LLM judge scores
- judge_summary.json           : aggregated LLM judge results

Trainer-based evaluation results are logged to Weights & Biases (W&B).

CONFIGURATION
-------------
- Paths and environment variables are managed via `.env`
- Model, dataset, and evaluation settings are defined in `config.yaml`

PORTABILITY
-----------
- Uses pathlib for cross-platform path handling
- Compatible with local environments, Docker, cloud VMs, and Colab
- No hardcoded paths; everything is config-driven

EXAMPLE
-------
Run full evaluation pipeline:

    python main_evaluation.py --mode generate
    python main_evaluation.py --mode lora
    python main_evaluation.py --mode judge
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from evaluation.output_generator import OutputGenerator
from evaluation.lora_metrics import LoRAMetrics
from evaluation.llm_judge import LLMJudge
from evaluation.trainer_metrics import TrainerMetrics

load_dotenv()


def resolve_path(path_str, base_dir):
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else base_dir / path


def run_generate(base_dir):
    config_path = resolve_path(os.getenv("CONFIG_PATH"), base_dir)
    eval_output_dir = resolve_path(os.getenv("EVAL_OUTPUT_DIR"), base_dir)

    generator = OutputGenerator(
        config_path=config_path,
        eval_dir=eval_output_dir
    )
    generator.run()


def run_lora(base_dir):
    generated_output_path = resolve_path(os.getenv("GENERATED_OUTPUT_PATH"), base_dir)
    eval_output_dir = resolve_path(os.getenv("EVAL_OUTPUT_DIR"), base_dir)

    metrics = LoRAMetrics(generated_output_path, eval_output_dir)
    summary = metrics.run()

    print("\nLoRA Evaluation Summary:\n", summary)


def run_judge(base_dir):
    config_path = resolve_path(os.getenv("CONFIG_PATH"), base_dir)
    eval_output_dir = resolve_path(os.getenv("EVAL_OUTPUT_DIR"), base_dir)

    judge = LLMJudge(
        config_path=config_path,
        eval_dir=eval_output_dir
    )
    summary = judge.run()

    print("\nLLM Judge Summary:\n", summary)


def run_trainer(base_dir):
    config_path = resolve_path(os.getenv("CONFIG_PATH"), base_dir)

    trainer = TrainerMetrics(config_path=config_path)
    trainer.run()


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "lora", "judge", "trainer"],
        required=True
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.mode == "generate":
        run_generate(base_dir)
    elif args.mode == "lora":
        run_lora(base_dir)
    elif args.mode == "judge":
        run_judge(base_dir)
    elif args.mode == "trainer":
        run_trainer(base_dir)


if __name__ == "__main__":
    main()
