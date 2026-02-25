import torch
import yaml
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login



# ---------- Load Config ----------
with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_id = config["model_id"]

# ---------- Load Dataset ----------
dataset = load_dataset(
    "json",
    data_files="data_pipeline/training_data.json",
    split="train",
)

dataset = dataset.remove_columns(["page_id"])


# -------- Train / Test Split --------
train_test = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

# -------- Train / Validation Split --------
train_val = train_test["train"].train_test_split(
    test_size=0.1,
    seed=42,
)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = train_test["test"]



# ---------- Login (Colab only if needed) ----------
# login()  

# ---------- W&B ----------
wandb.init(project=config["wandb"]["project"])

# ---------- Quantization Config ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---------- Load Model ----------
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# ---------- LoRA Config ----------
lora_cfg = config["lora"]

peft_config = LoraConfig(
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["alpha"],
    lora_dropout=lora_cfg["dropout"],
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# ---------- Training Arguments ----------
train_cfg = config["training"]

args = TrainingArguments(
    output_dir=train_cfg["output_dir"],
    per_device_train_batch_size=train_cfg["per_device_batch_size"],
    gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    num_train_epochs=train_cfg["epochs"],
    learning_rate=train_cfg["learning_rate"],
    weight_decay=train_cfg["weight_decay"],
    bf16=True,
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="wandb",
    push_to_hub=True,
    hub_model_id=config["hub"]["repo_id"],
    disable_tqdm=False,
)

# ---------- Formatting Function ----------
def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
    )

# ---------- Trainer ----------
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
    processing_class=tokenizer,
    formatting_func=formatting_func,
) 

trainer.train()

# ---------- Final Push ----------
trainer.push_to_hub()

wandb.finish()

 

# ----------  Evaluation on (training independent) Test Data ----------

test_metrics = trainer.evaluate(test_ds)

import json
with open("results/test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)