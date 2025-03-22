# Dockerfile

# Используем официальный Python-образ (3.12-slim)
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем список зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы приложения
COPY . .

# Устанавливаем PostgreSQL-клиент (libpq-dev), чтобы при необходимости работать с базой
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# Экспонируем порт, на котором работает Streamlit
EXPOSE 8501

# Запускаем приложение
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
