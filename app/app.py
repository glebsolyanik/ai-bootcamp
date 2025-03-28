import os
from dotenv import load_dotenv
import streamlit as st
from utils import db_utils

from ui.sidebar_ui import render_sidebar
from ui.chat_ui import render_chat

# Загружаем переменные окружения
load_dotenv()

# Параметры для БД
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Параметры для RAG
ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH")
ROUTER_CONFIG_PATH = os.environ.get("ROUTER_CONFIG_PATH")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
CLASSES_JSON_INFO_PATH = os.environ.get("CLASSES_JSON_INFO_PATH")
DATAFRAME_PATH = os.environ.get("DATAFRAME_PATH")


def main():

    if 'params_DB' not in st.session_state:
        st.session_state['params_DB'] = {
            "DB_NAME": DB_NAME,
            "DB_USER": DB_USER,
            "DB_HOST": DB_HOST,
            "DB_PORT": DB_PORT,
            "DB_PASSWORD": DB_PASSWORD
        }

    if 'params_RAG' not in st.session_state:
        st.session_state['params_RAG'] = {
            "ARTIFACTS_PATH": ARTIFACTS_PATH,
            "ROUTER_CONFIG_PATH": ROUTER_CONFIG_PATH,
            "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
            "CLASSES_JSON_INFO_PATH": CLASSES_JSON_INFO_PATH,
            "DATAFRAME_PATH": DATAFRAME_PATH
        }

    if 'LLM_agent' not in st.session_state:
        st.session_state['LLM_agent'] = None

    if 'selected_chat_id' not in st.session_state:
        st.session_state['selected_chat_id'] = None

    if 'chats' not in st.session_state:
        st.session_state['chats'] = None

    st.set_page_config(page_title="Мульти-чат с LLM", layout="wide")

    db_utils.create_tables()
    
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
