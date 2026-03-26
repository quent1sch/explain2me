"""
main_frontend_gradio_ui.py

Frontend chat interface using Gradio for Explain2Me.

Responsibilities:
- Display chat interface similar to ChatGPT
- Provide chat input box, send button, new chat button
- Allow switching between multiple chats (dropdown)
- Refresh chat list from backend
- Display chat history for selected chat
- Communicate with backend via FastAPI endpoints

Components:
------------
1. Dropdown: select existing chat
2. Refresh button: reload chat list from backend
3. Chatbot: display conversation (list of dicts {"role","content"})
4. Textbox: user message input
5. Send button: send message
6. New Chat button: start new conversation

Key Functions:
--------------
- fetch_chats(): get chat list from backend
- load_chat(chat_label): load selected chat + display history
- chat_fn(message, history): send message to backend + update history
- new_chat(): reset backend chat state
- refresh_chats(): refresh dropdown choices

Usage / Workflow:
-----------------
1. Start backend first:

    uvicorn main_backend_api:app --reload

2. Start frontend:

    python main_frontend_gradio_ui.py

3. Use UI:
    - Select chat from dropdown or create new chat
    - Type message and click Send
    - Refresh chat list if new chats created elsewhere

Notes / Recommendations:
------------------------
- Each message is sent via API to backend; Gradio waits for response
- Use streaming in future to make CPU responses appear progressively
- Chat history is restored when switching chats
- Backend DB is single source of truth (no local cache)
"""


import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"

# ----------------- HELPERS -----------------
def fetch_chats():
    res = requests.get(f"{API_URL}/chats").json()
    return {f"{c['chat_id']} - {c['title']}": c["chat_id"] for c in res}


def load_chat(chat_label):
    if not chat_label:
        return []

    chat_id = chats_map[chat_label]

    requests.post(f"{API_URL}/load_chat", params={"chat_id": chat_id})
    res = requests.get(f"{API_URL}/history").json()

    history = []
    for user_msg, assistant_msg in res["history"]:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

    return history


def chat_fn(message, history):
    res = requests.post(f"{API_URL}/chat", json={"text": message})
    reply = res.json()["response"]

    if history is None:
        history = []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return "", history


def new_chat():
    requests.post(f"{API_URL}/reset")
    return []


def refresh_chats():
    global chats_map
    chats_map = fetch_chats()
    return gr.update(choices=list(chats_map.keys()))


# ----------------- INITIAL LOAD -----------------
chats_map = fetch_chats()

# ----------------- UI -----------------
with gr.Blocks() as demo:
    gr.Markdown("# Explain2Me Chat")

    with gr.Row():
        dropdown = gr.Dropdown(
            choices=list(chats_map.keys()),
            label="Select Chat",
        )
        refresh_btn = gr.Button("Refresh")

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="Type your question...")

    with gr.Row():
        send = gr.Button("Send")
        new_btn = gr.Button("New Chat")

    # -------- EVENTS --------
    dropdown.change(load_chat, inputs=[dropdown], outputs=[chatbot])

    send.click(chat_fn, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[msg, chatbot])

    new_btn.click(new_chat, outputs=[chatbot])

    refresh_btn.click(refresh_chats, outputs=[dropdown])

# ----------------- LAUNCH -----------------
demo.launch()