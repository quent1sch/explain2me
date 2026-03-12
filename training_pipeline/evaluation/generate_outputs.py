"""
generate_outputs.py

Purpose
-------
Generate explanation outputs for a base LLaMA model and a LoRA adapter
for the Explain2Me project. Outputs are saved in a structured JSON file
to be later evaluated by an LLM-as-a-judge.

Outputs for each example include:
- Prompt / user request
- Reference explanation (test data)
- Base model output
- LoRA model output

This ensures a fair side-by-side comparison for downstream evaluation.
"""

import os
import json
import yaml
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------
# GET CONFIG
# ---------------------------

# BASE
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# dataset & split
hf_dataset_repo = config["dataset"]["hf_dataset_repo"]
test_split_ratio = config["dataset"]["test_split_ratio"]
seed = config["dataset"]["seed"]

# Model & LoRA
model_id = config["model_id"]
adapter_id = config["evaluation_trainer"]["adapter_id"]

# SUP.
NUM_SAMPLES = 10 # keep small
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
OUTPUT_FILE = "training_pipeline/evaluation/evaluation_results/generate_outputs.json"

# ---------------------------
# LOAD DATASET (test split)
# ---------------------------


dataset = load_dataset(hf_dataset_repo, split="train")
if "page_id" in dataset.column_names:
    dataset = dataset.remove_columns(["page_id"])

# Reproduce same splits as training
train_test = dataset.train_test_split(test_size=test_split_ratio, seed=seed)
test_ds = train_test["test"]


# ---------------------------
# TOKENIZER
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# QUANTIZATION (4-bit)
# ---------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------
# DEVICE
# ---------------------------

device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

# ---------------------------
# LOAD MODELS (evaluation mode)
# ---------------------------

# Base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
)
base_model.eval()

# LoRA model
lora_model = PeftModel.from_pretrained(base_model, adapter_id)
lora_model.eval()

# ---------------------------
# STRUCTURE FUNCTION FOR INFERENCE
# ---------------------------

def build_inference_prompt(messages):
    non_assistant_msgs = [m for m in messages if m["role"] != "assistant"]
    
    prompt = tokenizer.apply_chat_template(
        non_assistant_msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt

def get_content(messages, role):
    return next(m["content"] for m in messages if m["role"] == role)


# ---------------------------
# GENERATE OUTPUTS
# ---------------------------

os.makedirs("evaluation", exist_ok=True)
outputs = []

for example in tqdm(test_ds.select(range(NUM_SAMPLES))):

    # Build prompt from messages
    messages = example["messages"]

    prompt = build_inference_prompt(messages)
    question = get_content(messages, role="user")
    reference = get_content(messages, role="assistant")


    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        # Base model
        base_outputs = base_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE
        )
        # keep only generated part (not input prompt)
        generated_tokens = base_outputs[0][inputs["input_ids"].shape[-1]:]

        # decode generated text (without special tokens)
        base_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # LoRA model
        lora_outputs = lora_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE
        )
        generated_tokens = lora_outputs[0][inputs["input_ids"].shape[-1]:]
        lora_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    outputs.append({
        "question": question,
        "reference": reference,
        "base_output": base_text,
        "lora_output": lora_text,
    })

# ---------------------------
# SAVE OUTPUTS
# ---------------------------

with open(OUTPUT_FILE, "w") as f:
    json.dump(outputs, f, indent=2)

print(f"Generated {len(outputs)} examples saved to {OUTPUT_FILE}")