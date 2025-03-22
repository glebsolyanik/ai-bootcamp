import os
import streamlit as st
from chat.models.graph import ModelAgent
from chat.database.db import get_db_connection
def render_settings_ui():
    with st.sidebar:
        st.title("Настройки модели")
        
        model = st.text_input("Модель", value="llama-3-8b-instruct-8k")
        model_provider = st.text_input("Поставщик модели", value="openai")
        base_url = st.text_input("Базовый URL", value=os.environ.get("BASE_URL"))
        api_key = st.text_input("API ключ", value=os.environ.get("API_KEY"), type="password")
        
        if st.button("Сохранить"):
            if api_key:
                agent = ModelAgent(model, model_provider, base_url, api_key)
                if agent.validate_model():
                    st.session_state["agent"] = agent
                    st.session_state["history"] = get_db_connection()
                    st.success("Настройки сохранены")
                else:
                    st.error("Неверный API ключ")
            else:
                st.error("API ключ не может быть пустым")

        # if st.button("Удалить историю"):
        #     if "history" in st.session_state:
        #         st.session_state["history"].clear()
        #         st.session_state["chat_history"] = []
        #         st.success("История удалена")
        #     else:
        #         st.error("История не найдена")
        
        