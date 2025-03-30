# Используем базовый образ с Python (например, 3.10-slim)
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Сначала копируем только файл зависимостей для эффективного кэширования слоёв
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Предзагружаем модель SentenceTransformers для кэширования в образе
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Копируем сохранённые FAISS индексы из папки artifacts в образ
COPY artifacts /app/artifacts

# Копируем остальные файлы проекта
COPY . /app

# Открываем порт для Streamlit
EXPOSE 8501

# Запускаем приложение Streamlit с привязкой к 0.0.0.0 для доступа извне
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]

