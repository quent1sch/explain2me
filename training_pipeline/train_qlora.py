import os
import torch
import yaml
import wandb
from pathlib import Path
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import snapshot_download
from transformers.trainer_utils import get_last_checkpoint
from training_helpers import init_wandb

load_dotenv()


def train_lora(config_path: str = None):

    # ---------- Load Config ----------

    BASE_DIR = Path(__file__).resolve().parent

    config_path = Path(os.getenv("CONFIG_PATH", "../config.yaml"))

    if not config_path.is_absolute():
        config_path = (BASE_DIR.parent / config_path).resolve()

    # Load YAML config
    with config_path.open("r") as f:
        config = yaml.safe_load(f)


    model_id = config["model_id"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    repo_id = config["hub"]["repo_id"]
    hf_dataset_repo = config["dataset"]["hf_dataset_repo"]



    # ---------- Load Dataset ----------

    dataset = load_dataset(
        hf_dataset_repo,
        split="train"
    )

    if "page_id" in dataset.column_names:
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




    # ---------- Quantization Config ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ---------- Load Model & Tokenizer ----------
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)


    # ---------- LoRA Config ----------

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )


    # ---------- Training Arguments ----------

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=train_cfg["weight_decay"],
        bf16=True,
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

    # --------- Conditional Resume from HF Hub Checkpoint ---------

    resume_checkpoint = None

    try:
        print("\nChecking Hugging Face Hub for existing checkpoints.")

        # Download repo locally (cached)
        repo_path = snapshot_download(repo_id)

        # 1) try standard checkpoint detection (checkpoint-*)
        last_checkpoint = get_last_checkpoint(repo_path)

        # 2) fallback to last-checkpoint folder if no checkpoint-* found
        if resume_checkpoint is None:
            last_ckpt = os.path.join(repo_path, "last-checkpoint")
            if os.path.isdir(last_ckpt):
                print(f"Found 'last-checkpoint' on Hub. Resuming from: {last_ckpt}")
                resume_checkpoint = last_ckpt
            else:
                print("No checkpoint found on Hub. Training from scratch.")

        else:
            print(f"Resuming from last checkpoint: {resume_checkpoint}")

    except Exception as e:
        print("Could not detect Hub checkpoint. Starting fresh.")
        print(e)


    # ---------- W&B Initialize (or Resume) ----------
    # will resume from wandb id provided
    # else latest run in wandb if exisiting
    # else it wil create a new run
    # init_wandb() is a helper function for mapping depending on resuming or not
    # wandb.init(project=config["wandb"]["project"], id=run_id(str), resume="allow") 
    init_wandb(config, resume_checkpoint)



    # ---------- Train ----------
    # Auto resume safe:
    # trainer.train()
    trainer.train(resume_from_checkpoint=resume_checkpoint)


    # ---------- Final Push ----------
    trainer.push_to_hub()

    wandb.finish()

    

    # ----------  Evaluation on (training independent) Test Data ----------

    test_metrics = trainer.evaluate(test_ds)

    import json
    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    return trainer, test_metrics
