import os
from aiohttp import ClientSession

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


class SessionManager:
    _session: ClientSession | None = None

    @classmethod
    async def get_session(cls) -> ClientSession:
        """Возвращает сессию aiohttp, создавая её при первом вызове."""
        if cls._session is None or cls._session.closed:
            cls._session = ClientSession()
        return cls._session

    @classmethod
    async def close_session(cls):
        """Закрывает сессию, если она существует."""
        if cls._session is not None:
            await cls._session.close()
            cls._session = None


class Settings(BaseSettings):
    embedding_model: str = Field(json_schema_extra={".env": "EMBEDDING_MODEL"})
    rerank_model_name: str = Field(json_schema_extra={".env": "RERANK_MODEL_NAME"})
    base_dir: Path = Path(__file__).parent.parent.parent
    vector_store_dir: Path = base_dir / "vector_store"

    app_port: int = Field(json_schema_extra={".env": "APP_PORT"})
    app_url: str = Field(json_schema_extra={".env": "APP_URL"})
    app_host: str = Field(json_schema_extra={".env": "APP_HOST"})

    url_llm_model: str = Field(json_schema_extra={'.env': 'URL_LLM_MODEL'})
    api_key_llm_model: str = Field(json_schema_extra={'.env': 'API_KEY_LLM_MODEL'})
    api_token_telegram: str = Field(json_schema_extra={'.env': 'API_TOKEN_TELEGRAM'})

    log_config_path: str = Field(json_schema_extra={".env": "LOG_CONFIG_PATH"})

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
