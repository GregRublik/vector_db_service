from fastapi import FastAPI
from config.settings import logger, settings
import uvicorn
from api.v1.routers.vectordb import vectordb

app = FastAPI(
    title="VectorDB API",
    description="API для работы с векторной базой данных",
    version="0.1.0"
)

app.include_router(vectordb, prefix="/api/v1/vectordb")

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host=settings.app_host,
            port=settings.app_port,
            log_config=settings.log_config_path,
            use_colors=True,
            log_level="info",
            loop="asyncio"
        )
    except Exception as e:
        logger.error(f"Error launch app: {e}")
