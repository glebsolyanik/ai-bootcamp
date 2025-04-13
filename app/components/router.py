from langchain_core.messages import HumanMessage

from utils.prompts import router_instruction

from components.generate import BaseGenerator


from utils.state import State
from utils.prompts import router_instruction

from typing import Literal

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class RouteSchema(BaseModel):
    datasource_1: Literal["bank", "brave", "bridges_and_pipes", "documents", "dom_ru",
                        "edu", "FAQ_bulldog", "FAQ_skin", "gosuslugi", "Michelin", "potr_carz",
                        "rectifier", "red_mad_robot", "starvest", "telegram", "TK_RF", "world_class", "chitchat"] = Field(
        ...,
        description="""Given a user question, choose which datasource would be most 1 relevant for answering their question
        """,)
    datasource_2: Literal["bank", "brave", "bridges_and_pipes", "documents", "dom_ru",
                        "edu", "FAQ_bulldog", "FAQ_skin", "gosuslugi", "Michelin", "potr_carz",
                        "rectifier", "red_mad_robot", "starvest", "telegram", "TK_RF", "world_class", "chitchat"] = Field(
        ...,
        description="""Given a user question, choose which datasource would be most 2 relevant for answering their question
        """,
    )

class Router(BaseGenerator):
    def __init__(self, model, api_url, api_key) -> None:
        super().__init__(model, api_url, api_key)

        self.set_system_prompt(router_instruction)
        self.set_json_schema(RouteSchema)

    def route_query(self, state: State):
        result = self.generate_json_output(messages=[state['messages'][-1]])

        res = [result['datasource_1'], result['datasource_2']]

        return {"context_source": res}
