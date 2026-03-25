"""
main_backend_api.py

Backend server for Explain2Me chat application.

Responsibilities:
- Load and initialize Explain2MePipeline (model + memory)
- Provide FastAPI endpoints for chat, chat management, and history
- Persist chat messages in SQLite database
- Handle multiple chats and chat switching

Endpoints:
-----------
POST /chat
    Generate a reply to user message
    JSON input: {"text": "message", "new_chat": bool}
    Returns: {"response": "assistant reply"}

POST /reset
    Start a new chat (clears current chat state)

GET /chats
    List all chats from DB
    Returns: [{"chat_id": int, "title": str, "created_at": str}]

POST /load_chat
    Load a specific chat by chat_id
    Params: chat_id=int
    Returns: {"status": "loaded"}

GET /history
    Returns current chat history
    Format: [{"role": "user/assistant", "content": str}, ...]

Usage / Workflow:
-----------------
1. Start the server (development mode, autoreload):

    uvicorn main_backend_api:app --reload

2. Backend will load the model once (may take time on CPU/GPU)
3. API endpoints are ready for frontend calls (Gradio, JS, etc.)
4. Chat messages are persisted in SQLite DB (explain2me/inference/chat_history.db)

Notes:
------
- Model generation can be slow on CPU; consider reducing max_new_tokens for dev
- If multiple users connect, consider streaming generation for responsive UI
- current_chat_id auto-starts a new chat if None
"""

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import yaml, os
from dotenv import load_dotenv
from inference.chat_pipeline import Explain2MePipeline

# ----------------- SETUP -----------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

# ----------------- LOAD CONFIG -----------------
with open(BASE_DIR / "inference/chat_configs.yaml", "r") as f:
    inf_config = yaml.safe_load(f)

with open(BASE_DIR / "config.yaml", "r") as f:
    train_config = yaml.safe_load(f)

print("Loading model once and for all...")
pipeline = Explain2MePipeline.from_config(
    inf_config, train_config, hf_token=HF_TOKEN
)
print("Model ready!\nNext step: start UI with *python main_chat_gradio.py*")

# ----------------- API -----------------
app = FastAPI()

class Query(BaseModel):
    text: str
    new_chat: bool = False


# -------- CHAT --------
@app.post("/chat")
def chat(q: Query):
    if pipeline.current_chat_id is None:
        q.new_chat = True
    reply = pipeline.generate(q.text, is_new_chat=q.new_chat)
    return {"response": reply}


@app.post("/reset")
def reset():
    pipeline.current_chat_id = None
    pipeline.chat_history = []
    return {"status": "reset"}


# -------- CHAT MANAGEMENT --------
@app.get("/chats")
def get_chats():
    chats = pipeline.list_chats()
    return [
        {"chat_id": c[0], "title": c[1], "created_at": c[2]}
        for c in chats
    ]


@app.post("/load_chat")
def load_chat(chat_id: int):
    pipeline.load_chat(chat_id)
    return {"status": "loaded"}


@app.get("/history")
def get_history():
    history = []
    messages = pipeline.chat_history

    for i in range(len(messages)):
        if messages[i]["role"] == "user":
            user_msg = messages[i]["content"]
            assistant_msg = ""

            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                assistant_msg = messages[i+1]["content"]

            history.append((user_msg, assistant_msg))

    return {"history": history}