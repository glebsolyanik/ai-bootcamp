import os
import pickle

from semantic_router.routers import SemanticRouter

from utils.state import State

from outlines import models, generate
from typing import Literal

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant two datasource."""

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



class Router:
    def __init__(
            self,
            model, model_provider, api_url, api_key
    ) -> None:
        self.model = models.openai(
            model,
            base_url=api_url,
            api_key=api_key,
        )

        self.system = """You are an expert in directing user questions to the appropriate data source.
        Depending on the topic the question pertains to, select 1 relevant data source.
        bank — Оплата покупок бесконтактно, управление кредитами, открытие вкладов, переводы через приложение.
        brave — Вопросы о компании Brave Bison.
        bridges_and_pipes — Требования к проектированию мостов, труб и инженерных сооружений, включая расчёты, выбор материалов, защиту от
        коррозии и безопасность.
        documents — Описание СНИЛС и фискальных документов (чеки, БСО), их назначение, требования и применение.
        dom_ru — Услуги связи и управление балансом через приложение «Мой Дом.ру», кэшбэк, обещанный платёж, тарифы.
        edu — Специалитет и бакалавриат, новые образовательные структуры с 2026 года.
        FAQ_bulldog — Вопросы об уходе за бульдогами и жизни с ними.
        gosuslugi — Восстановление доступа, создание пароля, защита от мошенничества на портале Госуслуг.
        Michelin — Престижная награда ресторанам за качество кухни и сервиса, критерии оценки и поддержание звезды.
        potr_carz — Перечень товаров и услуг для оценки уровня доходов и социальных выплат в стране.
        rectifier — Финансовый отчёт компании Rectifier Technologies Ltd.
        red_mad_robot — Вопросы о компании Red Mad Robot.
        starvest — Вопросы о финансовом отчёте компании Starvest Plc.
        telegram — Вопросы о мессенджере Telegram.
        TK_RF — Трудовой кодекс РФ, трудовые отношения, права и обязанности сторон, условия труда.
        world_class — Мобильное приложение World Class 3.0, новые функции, управление расписанием и клубными услугами.
        chitchat - сообщение, которое ни относится ни к одному из вышеперечисленных классов."""

    def route_query(self, state: State):
        logger.warning(f"Message == {state["question"]}")
        message = self.system + "user's query:" + state["question"]
        generator = generate.json(self.model, RouteQuery)
        result = generator(
            message
        )

        res = [result.datasource_1, result.datasource_2]

        return {"context_source": res}
