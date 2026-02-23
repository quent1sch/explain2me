
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


class Settings:
    DB_NAME = "WikipediaOne.db"
    MAX_INPUT_TOKENS = 16384
    HF_BASE_URL = "https://router.huggingface.co/v1"

    # project root = folder where config.py lives
    BASE_DIR = Path(__file__).resolve().parent

    # if DB is inside data_pipeline/
    DB_PATH = BASE_DIR / "data_pipeline" / DB_NAME

    @staticmethod
    def get_db_path() -> str:
        return str(Settings.DB_PATH)


    @staticmethod
    def create_client() -> OpenAI:
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("HF_TOKEN environment variable not set.")
        
        return OpenAI(
            base_url=Settings.HF_BASE_URL,
            api_key=api_key
        )



