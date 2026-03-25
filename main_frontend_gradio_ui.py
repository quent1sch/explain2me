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