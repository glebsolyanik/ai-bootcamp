
from typing import TypedDict, List, Annotated, Sequence

from langchain.schema import BaseMessage

from langgraph.graph.message import add_messages

class State(TypedDict):
    question: str
    context_source: str
    context: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]