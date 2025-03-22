from dotenv import load_dotenv
import streamlit as st
from chat.ui.settings_ui import render_settings_ui
from chat.ui.chat_ui import render_chat_ui
from chat.database.db import init_db

load_dotenv()

def main():
    # Инициализируем базу данных при запуске
    db_initialized = init_db()
    if not db_initialized:
        st.error("Не удалось подключиться к базе данных. Проверьте переменные окружения.")
        return

    render_settings_ui()
    render_chat_ui()

if __name__ == "__main__":
    main() 