import streamlit as st
from dotenv import load_dotenv
from utils import (
    send_message_llm,
    init_db,
    load_history_db,
    save_history_db
)

load_dotenv()

# При запуске приложения инициализируем БД (создаём таблицу, если нет)
init_db()

st.title("🦙 Чат с LLaMA-3-8b")
st.caption("Используем OpenAI-совместимый API через LangChain")

# Подтягиваем историю из БД, если ещё не инициализирована
if "messages" not in st.session_state:
    st.session_state.messages = load_history_db(conversation_id="default")

# Кнопка очистки истории
if st.sidebar.button("🧹 Очистить историю"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    save_history_db(st.session_state.messages, conversation_id="default")

# Принимаем ввод пользователя
user_input = st.chat_input("Ваше сообщение...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Получаем ответ от LLM
    answer = send_message_llm(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Сохраняем обновлённую историю в БД
    save_history_db(st.session_state.messages, conversation_id="default")

