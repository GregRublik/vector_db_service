import os

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
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
    app_port: int = 8000
    app_host: str = "127.0.0.1"

    api_key_ranpod: str = Field(json_schema_extra={'.env': 'API_KEY_RANPOD'})
    api_token_telegram: str = Field(json_schema_extra={'.env': 'API_TOKEN_TELEGRAM'})

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# os.environ["HF_HOME"] = str(settings.base_dir / "hf_home")

prompt = """
Вы — AI-ассистент компании Ave Technologies, специализирующийся на предоставлении точной информации о компании на основе
 предоставленного документа. Ваши ответы должны соответствовать содержанию документа.

Правила взаимодействия:
1. Отвечайте ТОЛЬКО на основе предоставленного контекста
2. Если информация отсутствует в документе — отвечайте "В предоставленных материалах эта информация не указана"
3. Для запросов о компетенциях ссылайтесь на соответствующие разделы (Миссия, Ценности, Принципы работы)
"""