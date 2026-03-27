# Explain2Me

Explain2Me is an end-to-end NLP system for training and evaluating a **LoRA-adapted language model** that generates explanations tailored to a user’s level (age, education, job) — inferred directly from the input text.

The project covers the full ML lifecycle:
- data collection and dataset construction
- instruction tuning with **QLoRA**
- multi-step evaluation (automatic + LLM-based)
- inference via API and chat interface

The main focus is **parameter-efficient fine-tuning** and evaluation of the resulting model behavior.

Example input the model sees:

> *“I need to understand what a copula. I am a first-year bachelor student in statistics.”*

The model generates a clear, audience-aware explanation without requiring structured metadata — the user’s question alone guides the style, complexity, and structure of the response.

## Project Objective

**Explain2Me** aims to make knowledge accessible and tailored:

- Learns to simplify or enrich explanations depending on user-provided context

- Supports natural question phrasing — no separate attribute inputs required

- Adapts style and structure implicitly through fine-tuning

This is achieved by building a full instruction-based fine-tuning workflow for a language model, leveraging Wikipedia as a structured knowledge source.

---

## Problem

Standard instruction-tuned models generate generic explanations.

##### Goal:  
Train a model that adapts **style, complexity, and structure** based only on how the user asks the question.

No explicit metadata (no labels like “beginner” or “expert” at inference time).

---

## Approach

### Data → Training → Evaluation → Inference


### 1. Dataset construction (`main_data.py`)

- Scrapes Wikipedia pages to gather raw content

- Uses an LLM to generate simplified, audience-friendly explanations

- Formats data into chat-style instruction-response pairs (`system`, `user`, `assistant`)

- Produces a dataset ready for supervised LoRA fine-tuning

This transforms encyclopedic knowledge into high-quality, context-aware training data.
 
**Output:**
- training dataset (JSON)
- intermediate SQLite DB

---

#### 2. LoRA training (`main_lora_train.py`)

Implements a full **QLoRA training pipeline**:

- base instruct model + 4/8-bit quantization
- LoRA adapters (parameter-efficient fine-tuning)
- config-driven training (YAML)
- dataset splitting (train / val / test)
- checkpoint resume (Hugging Face Hub)
- experiment tracking (Weights & Biases)

**Training setup:**
- supervised fine-tuning on instruction dataset  
- base model frozen, only adapter weights updated  
- final adapter pushed to Hugging Face Hub  

**Run:**
```bash
python main_lora_train.py --config config.yaml
```


#### 3. Evaluation (main_evaluation.py)

Evaluation is modular and reproducible.

##### Step 1 — Generate outputs

```bash
python main_evaluation.py --mode generate
```

##### Step 2 — Local metrics

```bash
python main_evaluation.py --mode lora
```

- readability scoring
- ROUGE-L (overlap)
- BERTScore (semantic similarity)

##### Step 3 — LLM-as-a-judge

```bash
python main_evaluation.py --mode judge
```

- external model evaluates explanation quality


#### 4. Inference (main_backend_api.py)
- FastAPI backend
- streaming generation
- chat persistence (SQLite)
- multi-session support

Run (separately):
```bash
uvicorn main_backend_api:app --reload
```
```bash
python main_frontend_gradio_ui.py
```


### Project Structure
```
explain2me/
├── data_pipeline/
├── training_pipeline/
├── evaluation/
├── inference/
├── main_data.py
├── main_lora_train.py
├── main_evaluation.py
├── main_backend_api.py
├── main_frontend_gradio_ui.py
├── config.yaml
├── requirements.txt
├── README.md
└── LICENSE
```



## How to run
#### Setup
```bash
git clone https://github.com/quent1sch/explain2me.git
cd explain2me
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:
```bash
HF_TOKEN=...
OPENAI_API_KEY=...
WANDB_TOKEN=...
CONFIG_PATH=config.yaml
CHAT_CONFIG_PATH=inference/chat_configs.yaml
EVAL_OUTPUT_DIR=evaluation/evaluation_results
GENERATED_OUTPUT_PATH=evaluation/evaluation_results/generate_outputs.json
```

#### Full pipeline
1. Generate dataset
```bash
python main_data.py
```
2. Train LoRA adapter
```bash
python main_lora_train.py --config config.yaml
```
3. Evaluate
```bash
python main_evaluation.py --mode generate
python main_evaluation.py --mode lora
python main_evaluation.py --mode judge
```
4. Run inference API
```bash
uvicorn main_backend_api:app --reload
```
```bash
python main_frontend_gradio_ui.py
```

## Limitations
- training data partially synthetic (LLM-generated)
- evaluation relies partly on proxy metrics