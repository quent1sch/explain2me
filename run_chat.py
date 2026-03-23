"""
run_chat.py

Entry point for running the Explain2MePipeline in an interactive CLI session.

This script:
- Loads configuration files for inference and training
- Initializes the chat pipeline with optional Hugging Face credentials
- Provides a simple terminal-based chat interface
- Handles basic input/output and graceful error handling
"""

import sys
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv


# -------------------- PATH SETUP --------------------

BASE_DIR = Path(__file__).resolve().parent # script folder
# __file__ only exists if run as a script (no notebooks!)

REPO_ROOT = BASE_DIR.parent # main repo folder

# required for importing modules from the repo
sys.path.append(str(REPO_ROOT))


# -------------------- LOAD ENV --------------------
# 1. Try Colab secrets / environment variable
hf_token = os.environ.get("HF_TOKEN")

# 2. Try local .env if not found
if hf_token is None:
    try:
        load_dotenv(REPO_ROOT / ".env")
        HF_TOKEN = os.getenv("HF_TOKEN")
    except ImportError:
        pass

if not HF_TOKEN:
    print("Warning: Hugging Face token not found. Some models or summarization may fail.")

# -------------------- CONFIG PATHS --------------------

# Inference config
INF_CONFIG_PATH = REPO_ROOT / "inference" / "chat_configs.yaml"

# Training config (moved to main repo folder)
TRAIN_CONFIG_PATH = REPO_ROOT / "config.yaml"

with INF_CONFIG_PATH.open("r", encoding="utf-8") as f:
    inf_config = yaml.safe_load(f)

with TRAIN_CONFIG_PATH.open("r", encoding="utf-8") as f:
    train_config = yaml.safe_load(f)

# -------------------- INIT PIPELINE --------------------

from inference.chat_pipeline import Explain2MePipeline

pipeline = Explain2MePipeline.from_config(inf_config, train_config, hf_token=HF_TOKEN)

# -------------------- INTERACTIVE CHAT --------------------
print("Explain2Me Chat (type 'exit' or 'quit' to stop)\n")

while True:
    try:
        user_input = input("You: ")
    except EOFError:
        break

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break

    try:
        response = pipeline.generate(user_input)
        print(f"Assistant: {response}\n")
    except Exception:
        print("Assistant: Something went wrong.\n")