"""
output_generator.py

Generate explanation outputs for base and LoRA-adapted models.

PURPOSE
-------
This module generates explanations using both a base LLM and its LoRA-adapted
version on a held-out test dataset. Outputs are stored in a structured JSON file
for downstream evaluation (e.g., metrics or LLM-as-a-judge).

Outputs for each example include:
- Prompt / user request
- Reference explanation (test data)
- Base model output
- LoRA model output

This ensures a fair side-by-side comparison for downstream evaluation.

PORTABILITY
-----------
- Uses pathlib for cross-platform compatibility
- Supports .env configuration
- Works in local, Docker, cloud, and Colab environments

USAGE
-----
from evaluation.output_generator import OutputGenerator

gen = OutputGenerator(config_path="config.yaml")
gen.run()
"""

import os
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()


class OutputGenerator:
    """Generate outputs for base and LoRA models."""

    def __init__(self, config_path, eval_dir, num_samples=10):
        # ---------------------------
        # CONFIG
        # ---------------------------
        self.config_path = Path(config_path).resolve()

        with self.config_path.open("r") as f:
            self.config = yaml.safe_load(f)

        # ---------------------------
        # OUTPUT DIR (passed from main)
        # ---------------------------
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.eval_dir / "generated_outputs.json"

        # ---------------------------
        # CONFIG VALUES
        # ---------------------------
        self.model_id = self.config["model_id"]
        self.adapter_id = self.config["evaluation_trainer"]["adapter_id"]

        self.dataset_repo = self.config["dataset"]["hf_dataset_repo"]
        self.test_split_ratio = self.config["dataset"]["test_split_ratio"]
        self.seed = self.config["dataset"]["seed"]

        self.num_samples = num_samples
        self.max_new_tokens = 2048
        self.temperature = 0.7

        self.device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    # ---------------------------
    # DATA
    # ---------------------------
    def _load_dataset(self):
        dataset = load_dataset(self.dataset_repo, split="train")

        if "page_id" in dataset.column_names:
            dataset = dataset.remove_columns(["page_id"])

        train_test = dataset.train_test_split(
            test_size=self.test_split_ratio,
            seed=self.seed
        )

        self.test_ds = train_test["test"]

    # ---------------------------
    # MODELS
    # ---------------------------
    def _load_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device_map,
        )
        self.base_model.eval()

        self.lora_model = PeftModel.from_pretrained(
            self.base_model,
            self.adapter_id
        )
        self.lora_model.eval()

    # ---------------------------
    # PROMPT UTILS
    # ---------------------------
    def _build_prompt(self, messages):
        non_assistant_msgs = [m for m in messages if m["role"] != "assistant"]

        return self.tokenizer.apply_chat_template(
            non_assistant_msgs,
            tokenize=False,
            add_generation_prompt=True
        )

    @staticmethod
    def _get_content(messages, role):
        return next(m["content"] for m in messages if m["role"] == role)

    # ---------------------------
    # RUN
    # ---------------------------
    def run(self):
        print("Loading dataset...")
        self._load_dataset()

        print("Loading models...")
        self._load_models()

        outputs = []

        print("Generating outputs...")
        for example in tqdm(self.test_ds.select(range(self.num_samples))):

            messages = example["messages"]

            prompt = self._build_prompt(messages)
            question = self._get_content(messages, "user")
            reference = self._get_content(messages, "assistant")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)

            with torch.no_grad():
                # Base model
                base_out = self.base_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature
                )
                base_tokens = base_out[0][inputs["input_ids"].shape[-1]:]
                base_text = self.tokenizer.decode(base_tokens, skip_special_tokens=True)

                # LoRA model
                lora_out = self.lora_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature
                )
                lora_tokens = lora_out[0][inputs["input_ids"].shape[-1]:]
                lora_text = self.tokenizer.decode(lora_tokens, skip_special_tokens=True)

            outputs.append({
                "question": question,
                "reference": reference,
                "base_output": base_text,
                "lora_output": lora_text,
            })

        with self.output_file.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2)

        print(f"Generated {len(outputs)} samples → {self.output_file}")
        return outputs