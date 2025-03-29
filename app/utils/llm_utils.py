import logging
import os
import time

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from openai import OpenAIError
# from utils.RAG import RAG
from typing import TypedDict, List, Annotated, Sequence
from langchain.schema import Document, BaseMessage
from utils.vector_storage import VectorStorage
from langgraph.graph.message import add_messages


logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    context: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]

class LLMAgent:
    def __init__(self, model, model_provider, api_url, api_key):
        self.llm = init_chat_model(
            model=model,
            model_provider=model_provider,
            base_url=api_url,
            api_key=api_key,
        )

        self.vector_storage = VectorStorage(
            artifacts_path='artifacts/',
        )

        self.graph = self.create_graph()

    def validate_model(self):
        try:
            self.llm.invoke("test")
            return True
        except OpenAIError:
            return False

    def create_graph(self):
        builder = StateGraph(state_schema=State).add_sequence( [self.retrieve, self.generate])
        builder.add_edge(START, 'retrieve')

        graph = builder.compile(checkpointer=MemorySaver())
        return graph
    
    def retrieve(self, state: State):
        indexes = self.vector_storage.similarity_search(state["question"], 'bank')
        content = self.vector_storage.get_content(indexes)['answer'].to_list()
        return {"context": content}
    
    def generate(self, state: State):
        prompt = f"""Пользователь задал вопрос.
            Используя контекст, дай ему ответ на вопрос. Ответ должен быть емким, и опираться на контекст. 
    
            История переписки: {state["messages"]}
            
            Контекст: {state["context"]}
    
            Вопрос: {state["question"]}
            """
        response = self.llm.invoke(prompt)
        return {"messages": response}

    def send_message(self, messages, temperature, chat_id):

        self.llm.temperature = temperature

        stream =self.graph.invoke(
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
