"""
chat_pipeline.py

Core chat pipeline for Explain2Me conversational AI.

This module handles:
- Model loading (base + optional PEFT adapter)
- Chat generation using a structured prompt
- Conversation memory with summarization for long chats
- Persistent storage of chats and messages (SQLite)
- Optional Hugging Face API integration for summaries and titles
- Basic production safeguards (logging, retries, error handling, validation)
"""

from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import sqlite3
from huggingface_hub import InferenceClient
import logging
import time
import threading

# -------------------- LOGGING --------------------
# set log folder path at repo root and create it if needed
log_folder = Path(__file__).resolve().parent.parent / "logs"
log_folder.mkdir(parents=True, exist_ok=True)

log_file = log_folder / "chat_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=str(log_file), # find it in log folder
    filemode="a", # append to file
)

logger = logging.getLogger("Explain2Me")


# -------------------- EXPLAIN2ME CLASS --------------------
class Explain2MePipeline:
    def __init__(
    self,
    model_id,
    adapter_id=None,
    max_new_tokens=1024,
    temperature=0.7,
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
        self.summary_threshold = summary_threshold
        self.max_recent_messages = max_recent_messages
        self.summary_model = summary_model or "Qwen/Qwen2.5-7B-Instruct"
        self.stop_generation = False

        # Resolve db_path relative to the class file
        db_path = Path(db_path)
        if not db_path.is_absolute():
            db_path = Path(__file__).parent / db_path
        self.db_path = str(db_path)

        # Ensure parent folder exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

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
        self.conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=10
            )
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
        try:
            logger.info(f"Loading model: {self.model_id}")

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
                logger.info(f"Loading adapter: {self.adapter_id}")
                model = PeftModel.from_pretrained(base_model, self.adapter_id)
            else:
                model = base_model

            model.eval()
            logger.info("Model loaded successfully")
            return model

        except Exception:
            logger.exception("Model loading failed")
            raise RuntimeError("Failed to load model")

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

    
    # HF retry method for summary client
    def _safe_hf_call(self, messages, max_tokens=150, retries=3):
        if not self.summary_client:
            return None

        for attempt in range(retries):
            try:
                return self.summary_client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_tokens
                )
            except Exception:
                logger.warning(f"HF call failed (attempt {attempt+1})")
                time.sleep(2 ** attempt)

        logger.error("HF API failed after retries")
        return None

    # -------------------- CHAT MANAGEMENT --------------------
    
    def _new_chat(self, first_user_message=None):
        """Start a fresh conversation"""
        self.chat_history = [{"role": "system", "content": self.system_message}]
        title = self._generate_chat_title(first_user_message) if first_user_message else None

        c = self.conn.cursor()
        c.execute("INSERT INTO chats (title) VALUES (?)", (title,))
        self.conn.commit()
        self.current_chat_id = c.lastrowid

    def _generate_chat_title(self, first_message):
        """Generate a chat title"""
        if self.summary_client:
            try:
                messages = [{"role": "user", "content": f"Summarize this into a short 3-7 word title:\n{first_message}"}]
                response = self._safe_hf_call(messages)

                if response:
                    return response.choices[0].message["content"]
            
            except Exception:
                logger.exception("Title generation failed")
        
        return first_message[:25] + ("..." if len(first_message) > 25 else "")

    def load_chat(self, chat_id):
        try:
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

        except Exception:
            logger.exception("Failed to load chat")
            raise

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
        messages = [{"role": "user", "content": prompt}]
        completion = self._safe_hf_call(messages)

        if not completion:
            logger.warning("Skipping summarization due to HF failure")
            return
        
        summary = completion.choices[0].message["content"]

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
    
    def _generate_from_messages(self, streaming: bool = False):
        """
        - streaming=False → returns full string
        - streaming=True → yields tokens using TextIteratorStreamer
        """
        try:
            prompt = self.build_prompt()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            if not streaming:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                    )

                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # ---------------- STREAMING ----------------
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                streamer=streamer,
            )

            # Run generation in a separate thread
            thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
            )
            thread.start()

            # Yield tokens as they arrive
            for new_text in streamer:
                if self.stop_generation:
                    logger.info("Generation stopped by user")
                    break
                yield new_text

            thread.join()

        except Exception:
            logger.exception("Generation failed")
            if streaming:
                yield "Sorry, something went wrong while generating a response."
            else:
                return "Sorry, something went wrong while generating a response."


    
    def generate(self, user_prompt: str, is_new_chat: bool = False, streaming: bool = False):
        """
        Generate a reply to `user_prompt`.
        - If streaming=False: returns full text.
        - If streaming=True: yields tokens using TextIteratorStreamer
        """
        # Input validation
        if not isinstance(user_prompt, str) or not isinstance(is_new_chat, bool):
            return "Invalid input type."

        user_prompt = user_prompt.strip()
        if not user_prompt:
            return "Please enter a message."
        if len(user_prompt) > 5000:
            return "Message too long. Shorten it."
        
        try:
            self.stop_generation = False
            # Start new chat if needed
            if self.current_chat_id is None or is_new_chat:
                self._new_chat(first_user_message=user_prompt)

            self.chat_history.append({"role": "user", "content": user_prompt})
            self._store_message("user", user_prompt)

            if streaming:
                reply_buffer = ""

                try:
                    for chunk in self._generate_from_messages(streaming=True):
                        reply_buffer += chunk
                        yield chunk  # stream partial output
                    
                    if self.stop_generation:
                        logger.info("Stopped response not fully stored")

                except Exception:
                    logger.exception("Streaming generation failed")
                    yield "Something went wrong. Please try again."
                    return

                self.chat_history.append({"role": "assistant", "content": reply_buffer})
                self._store_message("assistant", reply_buffer)

            else:
                reply = self._generate_from_messages(streaming=False)
                self.chat_history.append({"role": "assistant", "content": reply})
                self._store_message("assistant", reply)
                return reply
        
        except Exception:
            logger.exception("Chat generation failed")

            if streaming:
                yield "Something went wrong. Please try again."
            else:
                return "Something went wrong. Please try again."



    # -------------------- DB WRITE --------------------
    def _store_message(self, role, content):
        if self.current_chat_id is None:
            logger.error("Attempted to store message, but no active chat id")
            return

        try:
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (self.current_chat_id, role, content),
            )
            self.conn.commit()

        except Exception:
            logger.exception("Writing message in DB failed")