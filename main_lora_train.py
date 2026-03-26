"""
# main_lora_train.py

LoRA Training Orchestrator

This script is the main entry point for training a LoRA-adapted language model
using the `train_qlora.py` module in the `training_pipeline` package.

What it does:
-------------
- Loads model, LoRA, and training configurations from YAML.
- Prepares datasets, including train/validation/test splits.
- Initializes the model with LoRA adapters and quantization settings.
- Handles checkpoint resuming from Hugging Face Hub.
- Integrates with Weights & Biases for logging and experiment tracking.
- Trains the model and pushes final weights to the hub.
- Evaluates the model on the test dataset and saves metrics.

Usage:
------
Run from the repository root:

    python main_lora_train.py --config path/to/config.yaml

- `--config` is optional. Defaults to `CONFIG_PATH` environment variable or `config.yaml` in the root.
- Designed for easy reuse, modularity, and clean package-based imports.
"""

import argparse

from training_pipeline.train_qlora import train_lora

def main():
    parser = argparse.ArgumentParser(description="LoRA Training Orchestration")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (optional, default uses CONFIG_PATH env or config.yaml)"
    )
    args = parser.parse_args()

    trainer, test_metrics = train_lora(config_path=args.config)
    print("\nTraining complete. Test metrics:")
    print(test_metrics)

if __name__ == "__main__":
    main()