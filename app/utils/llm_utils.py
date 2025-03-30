import logging
import time
import os
import pickle
import streamlit as st

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from openai import OpenAIError
from typing import TypedDict, List, Annotated, Sequence
from langchain.schema import BaseMessage
from utils.vector_storage import VectorStorage
from langgraph.graph.message import add_messages
from semantic_router.routers import SemanticRouter

logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    context: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: str

class LLMAgent:
    def __init__(self, model, model_provider, api_url, api_key):
        self.llm = init_chat_model(
            model=model,
            model_provider=model_provider,
            base_url=api_url,
            api_key=api_key,
        )

        artifacts_path = st.session_state['params_RAG']['ARTIFACTS_PATH']

        self.vector_storage = VectorStorage(
            artifacts_path=artifacts_path,
        )

        self.graph = self.create_graph()

        self.router = SemanticRouter.from_json(
            os.path.join(artifacts_path, st.session_state['params_RAG']['ROUTER_CONFIG_PATH']))
        with open(os.path.join(artifacts_path, st.session_state['params_RAG']['INDEX_ROUTER_PATH']), 'rb') as f:
            index = pickle.load(f)

        self.router.index = index

    def router(self, state: State):
        result = self.router(state["question"])
        return {"answer": result.name}

    def validate_model(self):
        try:
            self.llm.invoke("test")
            return True
        except OpenAIError:
            return False

    def create_graph(self):
        builder = StateGraph(state_schema=State).add_sequence([self.router, self.retrieve, self.generate])
        builder.add_edge(START, 'router')
        graph = builder.compile(checkpointer=MemorySaver())
        return graph

    def retrieve(self, state: State):
        # Если ответ уже равен "chitchat", не ищем контекст
        if state['answer'] is None or state['answer'] == "chitchat":
            return {"context": ""}
        indexes = self.vector_storage.similarity_search(state["question"], state['answer'])
        content = self.vector_storage.get_content(indexes)['answer'].to_list()
        return {"context": content}

    def generate(self, state: State):
        # Составляем prompt с жёсткими инструкциями:
        if state["context"] == "":
            prompt = f"""Общайся с пользователем.
История переписки: {state["messages"]}
Сообщение пользователя: {state["question"]}
"""
        else:
            # Внимательно прочитай ниже приведённый контекст и используй только его для ответа.
            # Если нужная информация отсутствует в контексте, ответь "Не знаю".
            prompt = f"""Ты отвечаешь на вопросы, основываясь только на приведённом ниже контексте. Не используй свои собственные знания.
Контекст:
{state["context"]}
История переписки: {state["messages"]}
Вопрос: {state["question"]}
Ответ:"""
        response = self.llm.invoke(prompt)
        return {"messages": response}

    def send_message(self, messages, temperature, chat_id):
        self.llm.temperature = temperature
        stream = self.graph.invoke(
            input={
                "question": messages[-1]['content'],
                "messages": messages
            },
            config={
                "configurable": {
                    "thread_id": chat_id,
                }
            },
            stream_mode="messages"
        )
        for msg, _ in stream:
            yield msg.content
            time.sleep(0.02)
