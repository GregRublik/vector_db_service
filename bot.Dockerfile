# Используем официальный образ Python
FROM python:3.12.4-slim

# Устанавливаем uv
RUN pip install uv

# Создаем рабочую директорию
WORKDIR /app

# Сначала копируем только файлы, необходимые для установки зависимостей
COPY pyproject.toml .

# Устанавливаем зависимости напрямую (без виртуального окружения в контейнере)
RUN uv pip install --system --group bot

# Копируем остальные файлы проекта
COPY src/bot.py src/config.py src/session_manager.py ./

## Запускаем оба сервиса
#CMD ["python src/bot.py"]