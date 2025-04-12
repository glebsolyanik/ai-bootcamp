import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph


from components.router import Router
from components.llm import LLM
from components.retriever import Retriever
from components.reranker import Reranker

from utils.trim_messages import trim_message_history
from utils.state import State

class RAGWorkflow:
    def __init__(
            self, 
            llm: LLM, 
            router: Router,
            retriever: Retriever,
            reranker: Reranker
        ) -> None:

        self.llm = llm
        self.router = router
        self.retriever = retriever
        self.reranker = reranker

        self.graph = self.create_graph()

    def create_graph(self):
        builder = StateGraph(state_schema=State).add_sequence([self.router.route_query, self.retriever.retrieve, self.reranker.bm25_reranker, self.llm.generate])
        builder.add_edge(START, 'route_query')

        graph = builder.compile(checkpointer=MemorySaver())
        return graph

    def send_message(self, messages, temperature, chat_id, d_descriptions_domen):
        self.llm.model.temperature = temperature

        messages = trim_message_history(messages)

        if len(messages) > 0:
            send_text = messages[-1]['content']
        else:
            send_text = "Скажи что не можешь ответить на это сообщение"

        stream = self.graph.stream(
            input={
                "question": send_text,
                "messages": messages,
                "d_descriptions_domen": d_descriptions_domen
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