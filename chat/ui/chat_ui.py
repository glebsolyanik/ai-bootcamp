import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chat.database.db import get_db_connection
from chat.models.graph import ModelAgent

def render_chat_ui():
    st.title(f"Чат с llm")

    if "agent" not in st.session_state:
        st.info("Перед началом работы, необходимо настроить модель")
    else:
        
        if "chat_history" not in st.session_state:        
            st.session_state["chat_history"] = []

            messages = st.session_state["history"].messages
            for message in messages:
                if isinstance(message, HumanMessage):
                    st.session_state.chat_history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    st.session_state.chat_history.append({"role": "assistant", "content": message.content})
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Введите ваше сообщение:"):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            st.session_state["history"].add_user_message(prompt)
        
            with st.chat_message("assistant"):
                result = st.session_state["agent"].graph.invoke({"messages": st.session_state["history"].messages}, {"configurable": {"thread_id": "abc123"}})
                response = result["messages"][-1].content
                st.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            st.session_state["history"].add_ai_message(response)
