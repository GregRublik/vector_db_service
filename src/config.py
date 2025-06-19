import os

from pathlib import Path

from pydantic_settings import BaseSettings
from loguru import logger

logger.add(
    "src/logs/debug.log",
    format="{time} - {level} - {message}",
    level="INFO",
    rotation="5 MB",
    compression="zip"
)

class Settings(BaseSettings):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    base_dir: Path = Path(__file__).parent.parent
    vector_store_dir: Path = base_dir / "vector_store"


settings = Settings()

os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# os.environ["HF_HOME"] = str(settings.base_dir / "hf_home")
