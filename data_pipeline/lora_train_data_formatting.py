import sqlite3
import random
from typing import List, Dict
import logging
import sys
from pathlib import Path
from config import Settings
import json
from pydantic import BaseModel, ValidationError


# Add parent folder to search path to import from config.py
parent_dir = Path().resolve().parent
print(parent_dir)
sys.path.append(str(parent_dir))


SYSTEM_PROMPT = (
    "You explain concepts to people at different age levels. "
    "Adapt explanations to the user (e.g. age, education, job)."
)

db_path = Settings.get_db_path()




# --------------------------------------------------
# Fetch data from DB
# --------------------------------------------------


def fetch_training_data(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT d.page_id, p.title, d.kind, d.content
        FROM definitions d
        JOIN pages p ON d.page_id = p.id
        WHERE d.kind IN ('simple', 'technical', 'kids')
    """)

    rows = cur.fetchall()
    conn.close()

    return rows




# --------------------------------------------------
# Audience generator based on kind
# --------------------------------------------------

def generate_user_prompt(title: str, kind: str) -> str:
    """
    Generate user request based on definition kind.
    """

    if kind == "kids":
        age = random.randint(6, 13)
        templates = [
            f"Can you explain to me what {title} is? I am {age} years old.",
            f"What is {title}? Please explain it for a {age}-year-old.",
            f"I'm {age}. Can you help me understand {title}?",
            f"I am in elementary school. What is {title}?",
            f"Can you explain {title} in very basic terms?"
        ]

    elif kind == "simple":
        ages = random.randint(12, 18)
        templates = [
            f"Explain {title} to a {ages}-year-old student.",
            f"What is {title}? I'm in high school.",
            f"Can you explain {title} in simple terms?"
        ]

    elif kind == "technical":
        roles = [
            "PhD student",
            "graduate student",
            "researcher",
            "engineer",
            "domain expert"
        ]
        role = random.choice(roles)

        templates = [
            f"What is {title}? Explain it to a {role}.",
            f"Provide a detailed explanation of {title} suitable for a {role}.",
            f"I am a {role}. Give me a technical explanation of {title}."
        ]

    else:
        templates = [f"Explain {title}."]

    return random.choice(templates)





# --------------------------------------------------
# Cleaning function for classic Wikipedia page content
# --------------------------------------------------

class Section(BaseModel):
    heading: str
    paragraphs: List[str]


def clean_content(content: str) -> str:
    try:
        raw = json.loads(content)
        sections = [Section(**item) for item in raw]
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise ValueError("Input must be a str representation of a JSON list of objects with keys 'heading' and 'paragraphs' corresponding to str, resp. List[str]") from e

    output_parts = []

    for section in sections:
        output_parts.append(f"[SECTION: {section.heading}]\n")

        for paragraph in section.paragraphs:
            output_parts.append(f"{paragraph}\n")

        output_parts.append("\n")

    return "\n".join(output_parts)






# --------------------------------------------------
# LoRA Adapter: training dataset builder
# --------------------------------------------------


def build_lora_training_dataset(db_path: str) -> List[Dict]:
    """
    Build a LoRA training dataset.
    Each row contains:
      - page_id
      - messages: list of system/user/assistant dicts
    """
    rows = fetch_training_data(db_path)

    training_data = []

    for page_id, title, kind, content in rows:
        try:
            # Generate user prompt
            user_prompt = generate_user_prompt(title, kind)

            # Clean content if not for kids
            if kind != 'kids':
                content = clean_content(content).strip()

            # Prepare assistant content
            assistant_content = f"[TITLE: {title}]\n\n{content}"

            # Prepare messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_content},
            ]

            training_data.append({
                "page_id": page_id,
                "messages": messages,
            })

        except Exception as e:
            logging.warning(f"Skipping page_id={page_id} due to error: {e}")
    
    return training_data






# --------------------------------------------------
# Store & Load LoRA training data
# --------------------------------------------------

training_data_path = "training_data.json"

def save_data(data, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_data(data_path):
    with open(data_path,"r") as file:
        test_training_data = json.load(file)
    return test_training_data



