
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file



load_dotenv()


class Settings:
    # -------------------- DATA --------------------
    DB_NAME = "WikipediaOne.db"
    TRAINING_DATA = "training_data.json"
    MAX_INPUT_TOKENS = 16384

    # -------------------- HUGGING FACE --------------------
    HF_BASE_URL = "https://router.huggingface.co/v1"
    HF_REPO_ID = "quent1sch/explain2me"
    HF_REPO_TYPE = "dataset"

    # -------------------- PATHS --------------------
    # project root = folder where config.py lives
    BASE_DIR = Path(__file__).resolve().parent

    # if DB is inside data_pipeline/
    DB_PATH = BASE_DIR / "data_pipeline" / DB_NAME
    TRAINING_DATA_PATH = BASE_DIR / "data_pipeline" / TRAINING_DATA

    # -------------------- UTILITIES --------------------
    @staticmethod
    def get_db_path() -> str:
        return str(Settings.DB_PATH)
    
    @staticmethod
    def get_training_data_path() -> str:
        return str(Settings.TRAINING_DATA_PATH)
    
    @staticmethod
    def get_hf_token() -> str:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN environment variable not set.")
        return token

    @staticmethod
    def create_client() -> OpenAI:
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("HF_TOKEN environment variable not set.")
        
        return OpenAI(
            base_url=Settings.HF_BASE_URL,
            api_key=api_key
        )

    @staticmethod
    def get_hf_api() -> HfApi:
        return HfApi(token=Settings.get_hf_token())


    # -------------------- PUSH TO HUB --------------------
    @staticmethod
    def push_dataset_to_hub(file_path: str = None):
        """
        Push a single file to Hugging Face Hub.
        Defaults to the training_data.json file.
        """

        file_path = file_path or Settings.get_training_data_path()

        api = Settings.get_hf_api()
        api.create_repo(
            repo_id=Settings.HF_REPO_ID,
            repo_type=Settings.HF_REPO_TYPE,
            exist_ok=True
        )

        upload_file(
            path_or_fileobj=file_path,
            path_in_repo="training_data.json",
            repo_id=Settings.HF_REPO_ID,
            repo_type=Settings.HF_REPO_TYPE,
            token=Settings.get_hf_token()
        )

        print(f"✅ File pushed successfully to Hugging Face Hub: {Settings.HF_REPO_ID}")