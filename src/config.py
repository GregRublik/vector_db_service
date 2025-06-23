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
    embedding_model: str = Field(json_schema_extra={".env": "EMBEDDING_MODEL"})
    base_dir: Path = Path(__file__).parent.parent
    vector_store_dir: Path = base_dir / "vector_store"

    app_port: int = Field(json_schema_extra={".env": "APP_PORT"})
    app_url: str = Field(json_schema_extra={".env": "APP_URL"})
    app_host: str = "0.0.0.0"

    url_llm_model: str = Field(json_schema_extra={'.env': 'URL_LLM_MODEL'})
    api_key_llm_model: str = Field(json_schema_extra={'.env': 'API_KEY_LLM_MODEL'})
    api_token_telegram: str = Field(json_schema_extra={'.env': 'API_TOKEN_TELEGRAM'})

    log_config_path: str = Field(json_schema_extra={".env": "LOG_CONFIG_PATH"})

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


prompt = """
Вы — AI-ассистент компании Ave Technologies, специализирующийся на предоставлении точной информации о компании на основе
 предоставленного документа. Ваши ответы должны соответствовать содержанию документа.

Правила взаимодействия:
1. Отвечайте ТОЛЬКО на основе предоставленного контекста
2. Если информация отсутствует в документе — отвечайте "В предоставленных материалах эта информация не указана"
3. Для запросов о компетенциях ссылайтесь на соответствующие разделы (Миссия, Ценности, Принципы работы)
"""
