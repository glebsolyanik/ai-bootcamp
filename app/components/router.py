from utils.state import State

from outlines import models, generate
from typing import Literal, List

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


def create_route_model_class(data_sources: List[str]):
    """
    Фабрика классов для создания RouteQuery с динамическими значениями Literal
    """
    # Создаем общий Literal тип для всех полей
    DataSourceLiteral = Literal[tuple(data_sources)]  # type: ignore

    class DynamicRouteQuery(BaseModel):
        """Route a user query to the most relevant two datasource."""

        datasource_1: DataSourceLiteral = Field(  # type: ignore
            ...,
            description="Given a user question, choose which datasource would be most 1 relevant"
        )

        datasource_2: DataSourceLiteral = Field(  # type: ignore
            ...,
            description="Given a user question, choose which datasource would be most 2 relevant"
        )

        class Config:
            arbitrary_types_allowed = True

    return DynamicRouteQuery


class Router:
    def __init__(
            self,
            model, api_url, api_key
    ) -> None:
        self.model = models.openai(
            model,
            base_url=api_url,
            api_key=api_key,
        )

        self.system = """You are an expert in directing user questions to the appropriate data source.
        Depending on the topic the question pertains to, select 2 relevant data source.
        """

    def route_query(self, state: State):
        logger.warning(f"Message == {state["question"]}")

        message = (self.system +
                   state['d_descriptions_domen']['descriptions'] +
                   "user's query:" + state["question"])

        # Создаем обновленную версию класса
        DynamicRouteQuery = create_route_model_class(state['d_descriptions_domen']['domens'])

        generator = generate.json(self.model, DynamicRouteQuery)
        result = generator(
            message, temperature=0
        )

        res = [result.datasource_1, result.datasource_2]

        return {"context_source": res}
