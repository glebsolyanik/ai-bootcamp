
from dataclasses import dataclass, field
from typing import TypedDict, List, Annotated, Sequence

from langchain.schema import BaseMessage

from langgraph.graph.message import add_messages

# @dataclass(kw_only=True)
# class State:
#     question:str = field(default=None)
#     is_rewrite_question:bool = field(default=False)
#     context_source: str = field(default=None)
#     context: List[str] = field(default=None)
#     messages: Annotated[Sequence[BaseMessage], add_messages] = field(default=None)
#     reflection_loop:int = field(default=0)


class State(TypedDict):
    question:str 
    context_source:str
    context:List[str]
    messages:Annotated[Sequence[BaseMessage], add_messages]
    reflection_loop:int
    is_need_reflection:bool
    d_descriptions_domen: dict
