"""
evaluate_lora_metrics.py

Portfolio-ready evaluation for Explain2Me LoRA adapter.

Metrics:
- ROUGE-L: alignment with structured reference outputs
- BERTScore: semantic fidelity
- Flesch-Kincaid grade: readability for target audience

Designed to be lightweight and runnable on Colab free.
"""

"""
evaluate_lora_metrics.py

Portfolio-ready evaluation for the Explain2Me LoRA adapter.

PURPOSE
This script evaluates a fine-tuned LoRA adapter of an instruction-tuned LLaMA model
for the Explain2Me project, which generates structured, simplified explanations 
conditional on the target user (e.g., kid, student, domain expert).

METRICS USED & WHY?
1. ROUGE-L: Measures overlap between the LoRA output and the reference (here, test data).
   Why? The dataset has a specific output structure; ROUGE-L quantifies how well the LoRA 
        model learns to adhere to this structure compared to the base model.

2. BERTScore: Computes SEMANTIC similarity between generated outputs and references.
   Why? LoRA may rephrase or restructure content. BERTScore ensures the meaning of the 
        explanation AND THE CONDITIONAL WORDING is preserved even if wording changes.

3. Flesch-Kincaid Grade: Evaluates readability/complexity of the generated text through
                         an assigned grade level.
   Why? Outputs must match the intended user type (e.g., child vs. expert).
        This metric provides an automatic check for audience-appropriate complexity, and
        will be used to compare correctness of audience targetting between base and lora.

DESIGN CONSIDERATIONS
- Lightweight & Colab-friendly: 4-bit quantization + no gradients for memory efficiency.
- Provides both quantitative metrics and qualitative example comparisons.
- Focused on LoRA-specific improvements: Highlights structural adherence, semantic fidelity,
  and readability improvements over the base LLaMA model.
"""

import os
import yaml
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
import bert_score
import textstat
from tqdm import tqdm



# ---------------------------
# CONFIG
# ---------------------------

# Load config
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

model_id = config["model_id"]
adapter_id = config["evaluation_trainer"]["adapter_id"]

hf_dataset_repo = config["dataset"]["hf_dataset_repo"]

test_split_ratio = config["dataset"]["test_split_ratio"]
seed = config["dataset"]["seed"]

NUM_EXAMPLES = 100  # limit for Colab free-tier
EVAL_OUTPUT_DIR = "evaluation"

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


# ---------------------------
# LOAD MODELS
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
lora_model = PeftModel.from_pretrained(base_model, adapter_id)


# ---------------------------
# LOAD TEST DATA
# ---------------------------

dataset = load_dataset(hf_dataset_repo, split="train")
dataset = dataset.remove_columns(["page_id"])
train_test = dataset.train_test_split(test_size=test_split_ratio, seed=test_split_ratio)
test_ds = train_test["test"][:NUM_EXAMPLES]


# ---------------------------
# GENERATE OUTPUTS
# ---------------------------

def generate_text(model, prompt, max_tokens=2048):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generating outputs...")
results = []

for example in tqdm(test_ds):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False)

    base_out = generate_text(base_model, prompt)
    lora_out = generate_text(lora_model, prompt)
    reference = tokenizer.apply_chat_template(example["messages"], tokenize=False)

    results.append({
        "prompt": prompt,
        "reference": reference,
        "base_output": base_out,
        "lora_output": lora_out
    })

# ---------------------------
# COMPUTE ROUGE-L
# ---------------------------

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def compute_rouge(preds, refs):
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(preds, refs)]
    return sum(scores) / len(scores)

base_rouge = compute_rouge([r["base_output"] for r in results], [r["reference"] for r in results])
lora_rouge = compute_rouge([r["lora_output"] for r in results], [r["reference"] for r in results])

# ---------------------------
# COMPUTE BERTScore
# ---------------------------

P, R, F1 = bert_score.score(
    [r["lora_output"] for r in results],
    [r["reference"] for r in results],
    lang="en"
)
lora_bertscore_f1 = F1.mean().item()

# ---------------------------
# COMPUTE FLESCH-KINCAID GRADE
# ---------------------------

def compute_readability(texts):
    return sum([textstat.flesch_kincaid_grade(t) for t in texts]) / len(texts)

lora_readability = compute_readability([r["lora_output"] for r in results])

# ---------------------------
# SAVE METRICS AND EXAMPLES
# ---------------------------

metrics = {
    "base_rougeL": base_rouge,
    "lora_rougeL": lora_rouge,
    "lora_bertscore_f1": lora_bertscore_f1,
    "lora_flesch_kincaid_grade": lora_readability,
    "rougeL_improvement": lora_rouge - base_rouge
}

with open(os.path.join(EVAL_OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(EVAL_OUTPUT_DIR, "sample_outputs.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete. Metrics saved to 'evaluation/' folder.")
print(metrics)