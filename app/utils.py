import os
import psycopg2
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI


load_dotenv()

def get_connection():
    conn = psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "mydatabase"),
        user=os.environ.get("DB_USER", "myuser"),
        password=os.environ.get("DB_PASSWORD", "mypassword"),
        host="db",
        port="5432",
        client_encoding='utf8'  # Добавьте этот параметр
    )
    return conn
def init_db():
    """
    Создаёт таблицу для хранения сообщений, если её нет.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(50),
                    role VARCHAR(50),
                    content TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

def load_history_db(conversation_id="default"):
    """
    Загружает историю сообщений из таблицы chat_messages.
    Возвращает список вида [{"role": "...", "content": "..."}].
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, content
                FROM chat_messages
                WHERE conversation_id = %s
                ORDER BY id ASC
            """, (conversation_id,))
            rows = cur.fetchall()

    if rows:
        return [{"role": r[0], "content": r[1]} for r in rows]
    else:
        # Если нет записей, возвращаем дефолтное system-сообщение
        return [{"role": "system", "content": "You are a helpful AI assistant."}]

def save_history_db(messages, conversation_id="default"):
    """
    Сохраняет текущую историю сообщений в таблицу chat_messages.
    Для упрощения: сначала удаляем старые сообщения, потом вставляем новые.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_messages WHERE conversation_id = %s", (conversation_id,))
            for msg in messages:
                cur.execute("""
                    INSERT INTO chat_messages (conversation_id, role, content)
                    VALUES (%s, %s, %s)
                """, (conversation_id, msg["role"], msg["content"]))

def send_message_llm(context_arr):
    """
    Формирует промпт из истории сообщений и отправляет запрос в LLM.
    Возвращает ответ как строку.
    """
    load_dotenv()

    API_URL = os.environ.get("API_URL", "https://llama3gpu.neuraldeep.tech/v1")
    API_KEY = os.environ.get("API_KEY", "")
    MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3-8b-instruct-8k")

    model = ChatOpenAI(
        openai_api_base=API_URL,
        model=MODEL_NAME,
        openai_api_key=API_KEY,
        temperature=0.7,
        max_tokens=350
    )

    prompt_text = ""
    for m in context_arr:
        prompt_text += f"Role:{m['role']} Content:{m['content']}\n"

    response = model.invoke(input=prompt_text)
    return response.content