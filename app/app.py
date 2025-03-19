from utils import send_message_llm
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME")
API_URL = os.environ.get("API_URL")
API_KEY = os.environ.get("API_KEY")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = API_KEY

st.title("Чат с LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Сообщение"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # send message to LLM
        ai_answer = send_message_llm(st.session_state.messages, API_URL, MODEL_NAME)
        st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
