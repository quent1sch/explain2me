"""
run_chat.py
Run Explain2MePipeline interactively, works in Colab or locally.
"""

import sys
import os
from pathlib import Path
import yaml


# -------------------- PATH SETUP --------------------
# __file__ only exists if run as a script (no notebooks!)
repo_root = Path(__file__).resolve().parents[0]

# required for importing modules from the repo
sys.path.append(str(repo_root))

from inference.chat_pipeline import Explain2MePipeline
 

# -------------------- LOAD HF TOKEN --------------------
# 1. Try Colab secrets / environment variable
hf_token = os.environ.get("HF_TOKEN")

# 2. Try local .env if not found
if hf_token is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root / ".env")
        hf_token = os.environ.get("HF_TOKEN")
    except ImportError:
        pass

if hf_token is None:
    print("Warning: Hugging Face token not found. Some models or summarization may fail.")

# -------------------- LOAD CONFIGS --------------------
with open(repo_root / "inference" / "chat_configs.yaml") as f:
    inf_config = yaml.safe_load(f)

with open(repo_root / "training_pipeline" / "config.yaml") as f:
    train_config = yaml.safe_load(f)

# -------------------- INIT PIPELINE --------------------

pipeline = Explain2MePipeline.from_config(inf_config, train_config, hf_token=hf_token)

# -------------------- INTERACTIVE CHAT --------------------
print("Explain2Me Chat (type 'exit' or 'quit' to stop)\n")

while True:
    try:
        user_input = input("You: ")
    except EOFError:
        # handles Colab cell end gracefully
        break

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break

    response = pipeline.generate(user_input)
    print(f"Assistant: {response}\n")