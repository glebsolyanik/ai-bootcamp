import os
import streamlit as st

from utils import db_utils
from utils.llm_utils import LLMAgent

def render_sidebar():
    with st.sidebar:
        render_model_settings()
        render_chat_list()

def render_chat_list():
    if st.session_state['LLM_agent'] is not None:
        st.header("Чаты")
        st.session_state['chats'] = db_utils.get_chats()

        if st.button("➕ Новый чат"):
            new_chat_id = db_utils.create_new_chat()
            st.session_state['selected_chat_id'] = new_chat_id
            st.session_state['chats'] = db_utils.get_chats()

            show_chat_list()
            st.rerun()

        show_chat_list()

def show_chat_list():
    if st.session_state['chats'] is not None and len(st.session_state['chats']) > 0:
        chat_names = [chat[1] for chat in st.session_state['chats']]

        current_chat_id = st.session_state['selected_chat_id']
        default_index = 0

        if current_chat_id is not None:
            for i, chat in enumerate(st.session_state['chats']):
                if chat[0] == current_chat_id:
                    default_index = i
                    break

        selected_index = st.radio(
            "Выберите чат:",
            options=range(len(chat_names)),
            format_func=lambda i: chat_names[i],
            index=default_index,
            key="chat_selector"
        )

        if isinstance(selected_index, int):
            new_selected_chat_id = st.session_state['chats'][selected_index][0]
        else:
            new_selected_chat_id = st.session_state['chats'][0][0]

        if current_chat_id != new_selected_chat_id:
            st.session_state['selected_chat_id'] = new_selected_chat_id
            st.rerun()
        else:
            st.session_state['selected_chat_id'] = new_selected_chat_id

        st.session_state['temperature'] = st.slider(
            "Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.01
        )
    else:
        st.write("У вас нет чатов. Создайте новый.")

def render_model_settings():
    """
    Удаляем ввод «Поставщик модели» и всегда используем openai (или другой при желании).
    """
    if st.session_state['LLM_agent'] is None:
        st.header("Настройки модели")

        model = st.text_input("Название модели", value=os.getenv("MODEL_NAME"))
        base_url = st.text_input("Базовый URL", value=os.getenv("API_URL"))
        api_key = st.text_input("API ключ", value=os.getenv("API_KEY"), type="password")

        if st.button("Сохранить настройки"):
            # Сохраняем в переменные окружения
            os.environ["MODEL_NAME"] = model
            os.environ["API_URL"] = base_url
            os.environ["API_KEY"] = api_key

            # Поставщик модели по умолчанию — openai (можно заменить при необходимости)
            model_provider = "openai"
            os.environ["MODEL_PROVIDER"] = model_provider

            # Создаём агента LLM
            LLM_agent = LLMAgent(
                os.getenv("MODEL_NAME"),
                os.getenv("MODEL_PROVIDER"),
                os.getenv("API_URL"),
                os.getenv("API_KEY"),
            )

            # Проверяем, что модель работает
            if LLM_agent.validate_model():
                st.session_state['LLM_agent'] = LLM_agent
                st.success("Модель успешно подключена")
                st.rerun()
            else:
                st.session_state['LLM_agent'] = None
                st.error("Не удалось подключиться к модели. Попробуйте изменить настройки")

