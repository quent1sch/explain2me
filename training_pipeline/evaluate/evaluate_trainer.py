# evaluate_trainer.py

import os
import torch
import yaml
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer


# ---------------------------
# CONFIG
# ---------------------------

# Load config
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Model & LoRA
model_id = config["model_id"]
adapter_id = config["evaluation_trainer"]["adapter_id"]

# HF Dataset
hf_dataset_repo = config["dataset"]["hf_dataset_repo"]

# W&B
wandb_project = config["wandb"]["project"] # same project as during training
wandb_group = config["evaluation_trainer"]["wandb_group"]
wandb_name = config["evaluation_trainer"]["wandb_name"]

# Evaluation split -> replicate train_test_split from train_qlora
test_split_ratio = config["dataset"]["test_split_ratio"]
seed = config["dataset"]["seed"]

# Training args (needed by SFTTrainer)
train_cfg = config["training"]
repo_id = config["hub"]["repo_id"]

# ---------------------------
# DEVICE
# ---------------------------

if torch.cuda.is_available():
    device_map = "auto"
else:
    device_map = {"": "cpu"}

# ---------------------------
# LOAD TOKENIZER
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# QUANTIZATION CONFIG
# ---------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    bnb_4bit_use_double_quant=True,  # improves quantization stability for 4-bit
)

# ---------------------------
# LOAD BASE MODEL
# ---------------------------

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
)

# ---------------------------
# LOAD LORA MODEL
# ---------------------------

lora_model = PeftModel.from_pretrained(
    base_model,
    adapter_id,
)

# ---------------------------
# LOAD DATASET
# ---------------------------

dataset = load_dataset(hf_dataset_repo, split="train")
if "page_id" in dataset.column_names:
    dataset = dataset.remove_columns(["page_id"])

# Reproduce same splits as training
train_test = dataset.train_test_split(test_size=test_split_ratio, seed=seed)
train_val = train_test["train"].train_test_split(test_size=test_split_ratio, seed=seed)
test_ds = train_test["test"]

# ---------------------------
# FORMATTING FUNCTION (same as training)
# ---------------------------

def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

# ---------------------------
# TRAINING ARGS (needed by SFTTrainer)
# ---------------------------

training_args = TrainingArguments(
    output_dir=train_cfg["output_dir"],
    per_device_train_batch_size=train_cfg["per_device_batch_size"],
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

# ---------------------------
# INIT W&B
# ---------------------------

wandb.init(
    project=wandb_project,
    group=wandb_group,
    name=wandb_name,
    job_type="evaluation"
)

# ---------------------------
# RECREATE TRAINER
# ---------------------------

trainer = SFTTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_val['train'], # required but we wont use it though
    eval_dataset=test_ds,
    peft_config=None, # already loaded
    processing_class=tokenizer,
    formatting_func=formatting_func,
)


# ---------------------------
# EVALUATE
# ---------------------------

print("Running evaluation on test split...")
test_metrics = trainer.evaluate(test_ds)


# ---------------------------
# LOG TO W&B
# ---------------------------

wandb.log(test_metrics)
wandb.finish()

print(
    f"""
    Evaluation complete. Metrics saved in WandB. 
    \nwandb_project = {wandb_project}
    \nwandb_group = {wandb_group}
    \nwandb_name = {wandb_name}
    \njob_type="evaluation"
    """
    )