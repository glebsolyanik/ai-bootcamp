import os
import psycopg2
from dotenv import load_dotenv
import streamlit as st
from utils.llm_utils import send_message_llm
from utils import db_utils

# Загружаем переменные окружения
load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME")
API_URL = os.environ.get("API_URL")
API_KEY = os.environ.get("API_KEY")

DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = API_KEY

# Инициализация приложения
st.set_page_config(page_title="Мульти-чат с LLM", layout="wide")
db_utils.create_tables(DB_NAME, DB_USER, DB_HOST, DB_PORT, DB_PASSWORD)
selected_chat_id = None

# Секция выбора чата и создания нового
with st.sidebar:
    st.header("Чаты")
    chats = db_utils.get_chats(DB_NAME, DB_USER, DB_HOST, DB_PORT, DB_PASSWORD)

    # Кнопка для создания нового чата
    if st.button("➕ Новый чат"):
        new_chat_id = db_utils.create_new_chat(DB_NAME, DB_USER, DB_HOST, DB_PORT, DB_PASSWORD)
        st.rerun()  # Перезагружаем страницу, чтобы отобразить новый чат

    # Если чаты есть, показываем список
    if chats:
        # Создаем список чатов и используем st.radio для выбора
        chat_options = {f"{chat[1]}": chat[0] for chat in chats}

        # Используем st.radio вместо st.selectbox
        selected_chat_label = st.radio("Выберите чат:", options=list(chat_options.keys()))

        # Получаем ID выбранного чата
        selected_chat_id = chat_options[selected_chat_label]

        # Добавляем возможность изменить название чата
        new_name = st.text_input("Изменить название чата", value=selected_chat_label.split(" (ID:")[0])

        if st.button("Сохранить изменения"):
            if new_name:
                db_utils.update_chat_name(selected_chat_id, new_name, DB_NAME, DB_USER, DB_HOST, DB_PORT, DB_PASSWORD)
                st.success(f"Название чата изменено на: {new_name}")
                st.rerun()  # Перезагружаем страницу, чтобы отобразить изменения

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.01)

    else:
        st.write("У вас нет чатов. Создайте новый.")

st.title("Чат с LLM")

if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = -1

# Загружаем сообщения для выбранного чата
if selected_chat_id is not None:
    st.session_state.chat_id = selected_chat_id
    st.session_state.messages = db_utils.load_chat_history(selected_chat_id, DB_NAME, DB_USER, DB_HOST, DB_PORT, DB_PASSWORD)

    # Отображаем сообщения
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Поле ввода
    if prompt := st.chat_input("Сообщение"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # Передаем параметры top_k, top_p, temperature в функцию инференса
            ai_answer = send_message_llm(st.session_state.messages, API_URL, MODEL_NAME, temperature)
            st.markdown(ai_answer)
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})

        # Сохраняем в базу
        db_utils.save_message_to_db(st.session_state.chat_id, "user", prompt, DB_NAME, DB_USER, DB_HOST, DB_PORT, DB_PASSWORD)
        db_utils.save_message_to_db(st.session_state.chat_id, "assistant", ai_answer, DB_NAME, DB_USER, DB_HOST, DB_PORT, DB_PASSWORD)
