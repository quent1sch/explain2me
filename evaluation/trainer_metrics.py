"""
trainer_metrics.py

Evaluation module for a LoRA-fine-tuned causal language model (e.g., LLaMA).

PURPOSE
-------
This class evaluates a LoRA-adapted model on a held-out test split, computing
loss and perplexity, and logs metrics to Weights & Biases (W&B).

KEY FEATURES
------------
- Config-driven setup (model, dataset, LoRA adapter, training settings, W&B).
- Loads base model in 4-bit (QLoRA) for memory-efficient inference.
- Applies LoRA adapter using PEFT.
- Recreates the same dataset splits used during training.
- Applies the model chat template and tokenizes evaluation data.
- Truncates long sequences (e.g., 2048 tokens) to avoid GPU OOM.
- Uses `prediction_loss_only=True` for memory-efficient evaluation.

PORTABILITY
-----------
- Uses pathlib for cross-platform paths
- Reads config paths from .env or defaults
- Works on local machine, Docker, cloud VM, or Colab

NOTES
-----
Training uses lazy tokenization (`tokenize=False`), while evaluation requires
pre-tokenized inputs (`input_ids`, `attention_mask`) for batching.

USAGE
-----
from evaluation.trainer_metrics import TrainerMetrics

trainer_metrics = TrainerMetrics(config_path="config.yaml")
metrics = trainer_metrics.run()
"""

import os
from pathlib import Path
import torch
import yaml
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()


class TrainerMetrics:
    """Self-contained evaluation for LoRA fine-tuned causal language models."""

    def __init__(self, config_path=None):
        base_dir = Path(__file__).resolve().parent

        self.config_path = Path(config_path or os.getenv("CONFIG_PATH"))
        if not self.config_path.is_absolute():
            self.config_path = base_dir / self.config_path
        self.config_path = self.config_path.resolve()

        with self.config_path.open("r") as f:
            self.config = yaml.safe_load(f)

        self.device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    def _load_tokenizer_model(self):
        model_id = self.config["model_id"]

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model with 4-bit quantization for memory efficiency
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
            ),
            device_map=self.device_map
        )

        # LoRA adapter
        adapter_id = self.config["evaluation_trainer"]["adapter_id"]
        self.model = PeftModel.from_pretrained(self.base_model, adapter_id)
        self.model.eval() # switch model to inference mode

    def _load_dataset(self):
        hf_dataset_repo = self.config["dataset"]["hf_dataset_repo"]
        dataset = load_dataset(hf_dataset_repo, split="train")
        if "page_id" in dataset.column_names:
            dataset = dataset.remove_columns(["page_id"])

        test_split_ratio = self.config["dataset"]["test_split_ratio"]
        seed = self.config["dataset"]["seed"]

        # reproduce same splits as during training
        train_test = dataset.train_test_split(test_size=test_split_ratio, seed=seed)
        train_val = train_test["train"].train_test_split(test_size=test_split_ratio, seed=seed)
        self.test_ds = train_test["test"]
        self.train_ds = train_val["train"]  # SFTTrainer requires a train dataset

    
    # preprocess/formating_func:
    # During training we set tokenize=False because SFTTrainer 
    # tokenizes lazily for memory efficiency.
    # Now in evaluation we must set tokenize=True so each example 
    # has 'input_ids' and 'attention_mask' 
    # before batching, otherwise SFTTrainer’s collator will throw a 
    # KeyError.
    
    # Plus, with colab free tier GPU, we get CUDA out of memory Error. 
    # This comes directly from the extensive length of messages in the dataset.
    # !!! Attention Mechanism Scales Quadratically with sequence length !!!
    # Therefore, for sake of memory, we truncate the sequence. But as dataset
    # has a specific role/content structure, we must preserve it.

    def _preprocess_example(self, example, max_tokens=1536):
        """Truncate long sequences and apply chat template for evaluation."""
        truncated_messages = []
        total_tokens = 0

        for msg in example["messages"]:
            role, content = msg["role"], msg["content"]
            tokens = self.tokenizer.encode(content)

            if total_tokens + len(tokens) > max_tokens:
                if role == "assistant":
                    remaining = max_tokens - total_tokens
                    if remaining > 0:
                        tokens = tokens[:remaining]
                        truncated_messages.append({"role": role, "content": self.tokenizer.decode(tokens)})
                    break
                else:
                    break
            else:
                truncated_messages.append({"role": role, "content": content})
                total_tokens += len(tokens)

        return self.tokenizer.apply_chat_template(truncated_messages, tokenize=True)

    def _preprocess_dataset(self):
        # Apply preprocessing function (truncate, chat format and tokenize) to the whole eval dataset
        self.test_ds = self.test_ds.map(
            self._preprocess_example,
            fn_kwargs={"max_tokens": 1536},
            remove_columns=self.test_ds.column_names
        )

    def _init_training_args(self):
        train_cfg = self.config["training"]
        repo_id = self.config["hub"]["repo_id"]
        self.training_args = TrainingArguments(
            output_dir=train_cfg["output_dir"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            num_train_epochs=train_cfg["epochs"],
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=train_cfg["weight_decay"],
            bf16=torch.cuda.is_available(),
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=train_cfg["eval_steps"],
            save_strategy="steps",
            save_steps=train_cfg["save_steps"],
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="wandb",
            push_to_hub=True,
            hub_model_id=repo_id,
            hub_strategy="checkpoint", # required for resume training: 
                                       # pushes checkpoint folders (not just model files like "every_save")
            remove_unused_columns=False, # required so evaluate() doesn't drop "messages" used by formatting_func
            disable_tqdm=False,
        )

    def _init_wandb(self):
        wandb_cfg = self.config["evaluation_trainer"]
        self.wandb_project = self.config["wandb"]["project"]
        wandb.init(
            project=self.wandb_project,
            group=wandb_cfg["wandb_group"],
            name=wandb_cfg["wandb_name"],
            job_type="evaluation"
        )

    def run(self):
        """Run full evaluation: dataset prep, evaluation, logging metrics to W&B."""
        print("Loading model and tokenizer...")
        self._load_tokenizer_model()

        print("Loading and preprocessing dataset...")
        self._load_dataset()
        self._preprocess_dataset()

        print("Initializing training arguments and W&B...")
        self._init_training_args()
        self._init_wandb()

        print("Creating trainer and evaluating...")
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_ds, # required but we wont use it though
            eval_dataset=self.test_ds,
            peft_config=None, # already loaded
            processing_class=None, # tokenizer and formatting done already with preprocess_eval
            formatting_func=None,
        )

        torch.cuda.empty_cache()
        with torch.no_grad():
            metrics = trainer.evaluate(self.test_ds)

        wandb.log(metrics)
        wandb.finish()

        print(f"Evaluation complete. Metrics logged to W&B project: {self.wandb_project}")

        return metrics