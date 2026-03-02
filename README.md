# Explain2Me: Overview

Explain2Me is an end-to-end ML project that teaches a language model to explain complex concepts in a way that adapts to the user’s background, experience, and context — based solely on how the user phrases their question.

Example input the model sees:

> *“I need to understand what a copula. I am a first-year bachelor student in statistics.”*

The model generates a clear, audience-aware explanation without requiring structured metadata — the user’s question alone guides the style, complexity, and structure of the response.


## Project Objective

**Explain2Me** aims to make knowledge accessible and tailored:

- Learns to simplify or enrich explanations depending on user-provided context

- Supports natural question phrasing — no separate attribute inputs required

- Adapts style and structure implicitly through fine-tuning

This is achieved by building a full instruction-based fine-tuning workflow for a language model, leveraging Wikipedia as a structured knowledge source.

## Technical Overview

### Data Pipeline (Completed)

- Scrapes Wikipedia pages to gather raw content

- Uses an LLM to generate simplified, audience-friendly explanations

- Formats data into chat-style instruction-response pairs (`system`, `user`, `assistant`)

- Produces a dataset ready for supervised LoRA fine-tuning

This transforms encyclopedic knowledge into high-quality, context-aware training data.

### LoRA Fine-Tuning (In Progress)

- Fine-tunes a base instruct model using the generated dataset

- Uses a LoRA adapter for parameter-efficient training

- Learns to generate explanations conditioned implicitly on user-provided context

Currently: the LoRA adapter is being trained to respond appropriately to diverse user questions.

### Inference Pipeline (Next Step)

- Users will input natural questions including their context

- The model generates personalized, audience-aware explanations

- Leverages the LoRA adapter to condition output based on phrasing and implied attributes


## Projects Key Points

- End-to-end ML pipeline: data collection → dataset engineering → fine-tuning → inference

- Instruction-based, chat-style dataset creation

- Parameter-efficient adaptation using LoRA

- Demonstrates applied skills in NLP, LLM adaptation, and prompt-driven data generation

- Focused on making explanations personalized and accessible

🗂 Project Structure
```
explain2me/
├── data_pipeline/                 # Core scraping + dataset creation logic
├── config.py                      # Project settings and paths
├── main_data.py                   # Entry point to run the data pipeline
├── requirements.txt               # Python dependencies
├── README.md
└── LICENSE
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/quent1sch/explain2me.git
cd explain2me
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# or
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The project uses environment variables for sensitive tokens:

```python
HF_TOKEN=<your_HF_token>
OPENAI_API_KEY=<your_OPENAI_key>
WANDB_TOKEN=<your_wandb_token>
```

Set it in your shell before running.

## Usage

Run the full data pipeline to generate the training dataset:
```bash
python main_data.py
```

This will:

1. Scrape Wikipedia pages from seed URLs

2. Generate simplified, audience-aware explanations via an LLM

3. Format the dataset for LoRA fine-tuning

4. Optionally push the dataset to the Hugging Face Hub

## Output

- Training dataset: training_data.json (chat-style format)

- LoRA adapter: trained weights for fine-tuning the base instruct model

Once the inference pipeline is implemented, users will be able to input natural questions and receive personalized explanations.

## Tips

- Make sure Wikipedia URLs are listed in `data_pipeline/wiki_urls` before running

- Logs progress in `data_pipeline_logs.log` for monitoring pipeline execution

## License

This project is licensed under the Apache-2.0 License — see `LICENSE` for details.