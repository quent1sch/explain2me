import torch
import json
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm


# ---------------------------
# CONFIG
# ---------------------------

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_id = "your-username/llama-3.1-8b-explain2me-lora"

wandb_project = "explain2me-evaluation"
num_generation_samples = 5


# ---------------------------
# INIT W&B
# ---------------------------

wandb.init(project=wandb_project)


# ---------------------------
# LOAD DATA (TEST SPLIT ONLY)
# ---------------------------

# Reproduce same split as training
dataset = load_dataset(
    "json",
    data_files="data_pipeline/training_data.json",
    split="train",
)

dataset = dataset.remove_columns(["page_id"])

train_test = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

train_val = train_test["train"].train_test_split(
    test_size=0.1,
    seed=42,
)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = train_test["test"]



# ---------------------------
# LOAD TOKENIZER
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# ---------------------------
# QUANT CONFIG
# ---------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ---------------------------
# LOAD BASE MODEL
# ---------------------------

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)


# ---------------------------
# LOAD LORA MODEL
# ---------------------------

lora_model = PeftModel.from_pretrained(
    base_model,
    adapter_id
)


# ---------------------------
# LOSS EVALUATION FUNCTION
# ---------------------------

def compute_loss(model, dataset):
    model.eval()
    total_loss = 0

    for example in tqdm(dataset):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()

    return total_loss / len(dataset)


print("Evaluating Base Model...")
base_loss = compute_loss(base_model, test_ds)

print("Evaluating LoRA Model...")
lora_loss = compute_loss(lora_model, test_ds)


# ---------------------------
# GENERATION COMPARISON
# ---------------------------

generation_samples = []

for i in range(num_generation_samples):
    example = test_ds[i]
    messages = example["messages"]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)

    with torch.no_grad():
        base_output = base_model.generate(**inputs, max_new_tokens=200)
        lora_output = lora_model.generate(**inputs, max_new_tokens=200)

    generation_samples.append({
        "prompt": prompt,
        "base_output": tokenizer.decode(base_output[0], skip_special_tokens=True),
        "lora_output": tokenizer.decode(lora_output[0], skip_special_tokens=True),
    })


# ---------------------------
# SAVE RESULTS
# ---------------------------

results = {
    "base_loss": base_loss,
    "lora_loss": lora_loss,
    "loss_improvement": base_loss - lora_loss,
}

with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("evaluation/sample_outputs.json", "w") as f:
    json.dump(generation_samples, f, indent=2)


# ---------------------------
# LOG TO W&B
# ---------------------------

wandb.log(results)

wandb.finish()

print("Evaluation complete.")
print(results)