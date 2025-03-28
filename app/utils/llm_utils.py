from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from openai import OpenAIError
from utils.RAG import RAG
import logging
import os
import time

logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self, model,
                 model_provider,
                 api_url,
                 api_key,
                 router_config_path: str,
                 embedding_model_name: str,
                 classes_json_info_path: str,
                 artifacts_path: str,
                 dataframe_path: str
                 ):
        self.llm = init_chat_model(
            model=model,
            model_provider=model_provider,
            base_url=api_url,
            api_key=api_key,
        )

        self.graph = self.create_graph()

        self.rag = RAG(router_config_path=os.path.join(artifacts_path, router_config_path),
                       embedding_model_name=embedding_model_name,
                       classes_json_info_path=os.path.join(artifacts_path, classes_json_info_path),
                       artifacts_path=artifacts_path,
                       dataframe_path=os.path.join(artifacts_path, dataframe_path))

    def call_model(self, state: MessagesState):
        response = self.llm.invoke(state["messages"])
        return {"messages": response}

    def create_graph(self):
        builder = StateGraph(state_schema=MessagesState)
        builder.add_node('call_model', self.call_model)
        builder.add_edge(START, 'call_model')
        graph = builder.compile(checkpointer=MemorySaver())
        return graph

    def validate_model(self):
        try:
            self.llm.invoke("test")
            return True
        except OpenAIError:
            return False

    def send_message(self, messages, temperature, chat_id):

        self.llm.temperature = temperature

        result_rag = self.rag.process(messages[-1]['content'])

        if len(result_rag) > 0:
            search_answer = result_rag.iloc[0]['answer']

            prompt = f"""Пользователь задал вопрос.
            Используя контекст, дай ему ответ на вопрос. Ответ должен быть емким, и опираться на контекст. 
    
            Контекст: {search_answer}
    
            Вопрос: {messages}
            """

            messages[-1] = {"role": "user", "content": prompt}

        stream = self.graph.stream(
            input={"messages": messages},
            config={
                "configurable": {
                    "thread_id": chat_id,
                }
            },
            stream_mode="messages"
        )

        for msg, metadata in stream:
            yield msg.content
            time.sleep(0.02)
