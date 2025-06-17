import os

from pathlib import Path

from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).parent.parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HOME"] = str(BASE_DIR / "hf_home")

class Settings(BaseSettings):
    pass


settings = Settings()
