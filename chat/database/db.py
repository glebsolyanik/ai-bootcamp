import os
from dotenv import load_dotenv
import psycopg2
from langchain_community.chat_message_histories import (
    PostgresChatMessageHistory,
)

load_dotenv()

def init_db():
    """Инициализация базы данных и создание необходимых таблиц"""
    conn_string = f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
    try:
        # Создаем соединение
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        conn.autocommit = True
        
        # Создаем курсор
        cursor = conn.cursor()
        
        # Создаем таблицу для хранения сообщений, если она не существует
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS langchain_chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            message JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.close()
        return True
    except Exception as e:
        print(f"Ошибка при инициализации базы данных: {e}")
        return False

def get_db_connection():
    """Получение подключения к базе данных для сохранения истории чата"""
    # Сначала инициализируем БД, если это первый запуск
    init_db()
    
    connection_string = f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
    
    # Создаем и возвращаем историю чата
    return PostgresChatMessageHistory(
        connection_string=connection_string,
        session_id="default_session",
        table_name="langchain_chat_history"  # Указываем название таблицы
    )








