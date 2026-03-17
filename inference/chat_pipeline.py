"""
chat_pipeline.py
"""

from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sqlite3
from huggingface_hub import InferenceClient







# -------------------- EXPLAIN2ME CLASS --------------------
class Explain2MePipeline:
    def __init__(
    self,
    model_id,
    adapter_id=None,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    load_in_4bit=True,
    summary_threshold=2000, # max tokens before summarization
    max_recent_messages=6, # when summarize chat history, keep last 6 messages intact
    system_message=None, # auto system message set below
    db_path="chat_history.db",
    hf_token=None,
    summary_model=None,
    ):

        self.model_id = model_id
        self.adapter_id = adapter_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.db_path = db_path
        self.summary_threshold = summary_threshold
        self.max_recent_messages = max_recent_messages
        self.summary_model = summary_model or "Qwen/Qwen2.5-7B-Instruct"

        # tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model(load_in_4bit)

        self.system_message = system_message or (
            "You explain concepts clearly and adapt explanations "
            "to the user's level (age, education, job)."
            )

        self.chat_history = []
        self.current_chat_id = None

        # DB 
        # connect to DB and initialize it if does not exist
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()


        # Summary client (separate model) for summarization of chats & titles
        self.summary_client = (
            InferenceClient(model=self.summary_model, token=hf_token)
            if hf_token else None
            )
    
    # -------------------- CONFIG --------------------
    @classmethod
    def from_config(cls, inf_config, train_config, hf_token=None):
        return cls(
            model_id=train_config["model_id"],
            adapter_id=train_config["hub"]["repo_id"],

            # generation
            max_new_tokens=inf_config["generation"]["max_new_tokens"],
            temperature=inf_config["generation"]["temperature"],

            # memory
            summary_threshold=inf_config["memory"]["summary_threshold"],
            max_recent_messages=inf_config["memory"]["max_recent_messages"],

            # system + db
            system_message=inf_config["system_prompt"],
            db_path=inf_config["chat_history"]["db_path"],

            # summarization
            summary_model=inf_config["summarization"]["model"],
            hf_token=hf_token,
        )

    # -------------------- MODEL LOADING --------------------
    def _load_model(self, load_in_4bit):
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                )

        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            )

        if self.adapter_id:
            model = PeftModel.from_pretrained(base_model, self.adapter_id)
        else:
            model = base_model

        model.eval()
        return model

    # -------------------- DATABASE --------------------
    def _init_db(self):
        c = self.conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
        )
        """)

        self.conn.commit()

    # -------------------- CHAT MANAGEMENT --------------------
    def new_chat(self, first_user_message=None):
        """Start a fresh conversation"""
        self.chat_history = [{"role": "system", "content": self.system_message}]

        title = (
            self._generate_chat_title(first_user_message)
            if first_user_message else None
            )

        c = self.conn.cursor()
        c.execute("INSERT INTO chats (title) VALUES (?)", (title,))
        self.conn.commit()
        self.current_chat_id = c.lastrowid

    def _generate_chat_title(self, first_message):
        """Generate a chat title"""
        if self.summary_client:
            prompt = f"Summarize this into a short 3-7 word title:\n{first_message}"
            return self.summary_client.text_generation(
                prompt, max_new_tokens=30, stop=["\n"]
                ).strip()
        return first_message[:25] + ("..." if len(first_message) > 25 else "")

    def load_chat(self, chat_id):
        c = self.conn.cursor()
        c.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp",
            (chat_id,),
            )

        messages = [{"role": r, "content": c} for r, c in c.fetchall()]

        if not messages:
            raise ValueError(f"No chat found with chat_id={chat_id}")

        self.chat_history = [{"role": "system", "content": self.system_message}] + messages
        self.current_chat_id = chat_id

    def list_chats(self):
        c = self.conn.cursor()
        c.execute("SELECT chat_id, title, created_at FROM chats ORDER BY created_at DESC")
        return c.fetchall()

    # -------------------- TOKEN COUNT --------------------
    def _token_count(self):
        text = "\n".join(m["content"] for m in self.chat_history)
        return len(self.tokenizer(text)["input_ids"])

    # -------------------- SUMMARIZATION --------------------
    def _summarize_old_messages(self):
        """Summarize old messages (up to the last {self.max_recent_messages})"""
        if not self.summary_client:
            return

        # avoid summarizing multiple times
        if any("Chat summary:" in m["content"] for m in self.chat_history):
            return

        to_summarize = self.chat_history[1:-self.max_recent_messages]

        if not to_summarize:
            return

        combined_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_summarize
            )

        prompt = f"""
            Summarize the following conversation in 3-6 sentences.

            Focus on:
            1. Concepts the user wants to understand
            2. Explanations already given
            3. Open questions

            Conversation:
            {combined_text}
            """

        summary = self.summary_client.text_generation(
            prompt, max_new_tokens=150
            ).strip()

        # rebuild memory
        self.chat_history = (
            [self.chat_history[0]]
            + [{"role": "system", "content": f"Chat summary: {summary}"}]
            + self.chat_history[-self.max_recent_messages:]
            )

        # store summary
        c = self.conn.cursor()
        c.execute(
            "UPDATE chats SET summary=? WHERE chat_id=?",
            (summary, self.current_chat_id),
        )
        self.conn.commit()

    # -------------------- MEMORY LAYER --------------------
    def _get_prompt_messages(self):
        system = []
        summary = []
        recent = []

        for m in self.chat_history:
            if m["role"] == "system" and "Chat summary:" in m["content"]:
                summary.append(m)
            elif m["role"] == "system":
                system.append(m)

        non_system = [m for m in self.chat_history if m["role"] != "system"]
        recent = non_system[-self.max_recent_messages:]

        return system + summary + recent

    # -------------------- PROMPT --------------------
    def build_prompt(self):
        if self._token_count() > self.summary_threshold:
            self._summarize_old_messages()

        messages = self._get_prompt_messages()

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # -------------------- GENERATION --------------------
    def _generate_from_messages(self):
        prompt = self.build_prompt()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def generate(self, user_prompt):
        if self.current_chat_id is None:
            self.new_chat(first_user_message=user_prompt)
        
        # Add user message
        self.chat_history.append({"role": "user", "content": user_prompt})
        self._store_message("user", user_prompt)

        # Generate assistant reply
        reply = self._generate_from_messages()

        self.chat_history.append({"role": "assistant", "content": reply})
        self._store_message("assistant", reply)

        return reply

    # -------------------- DB WRITE --------------------
    def _store_message(self, role, content):
        if self.current_chat_id is None:
            raise RuntimeError("No active chat.")

        c = self.conn.cursor()
        c.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (self.current_chat_id, role, content),
        )
        self.conn.commit()