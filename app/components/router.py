import os
import pickle

from semantic_router.routers import SemanticRouter

from utils.state import State

from langchain.chat_models import init_chat_model
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field



class MyRouteQuery(BaseModel):
    """Route a user query to the most relevant two datasource."""

    datasource_1: Literal["bank", "brave", "bridges_and_pipes", "documents", "dom_ru",
    "edu", "FAQ_bulldog", "FAQ_skin", "gosuslugi", "Michelin", "potr_carz",
    "rectifier", "red_mad_robot", "starvest", "telegram", "TK_RF", "world_class"] = (Field(
        ...,
        description="""Given a user question, choose which datasource would be most 1 relevant for answering their question
        """, ))
    datasource_2: Literal["bank", "brave", "bridges_and_pipes", "documents", "dom_ru",
    "edu", "FAQ_bulldog", "FAQ_skin", "gosuslugi", "Michelin", "potr_carz",
    "rectifier", "red_mad_robot", "starvest", "telegram", "TK_RF", "world_class"] = Field(
        ...,
        description="""Given a user question, choose which datasource would be most 2 relevant for answering their question
        """,
    )


class Router:
    def __init__(self, llm) -> None:

        structured_llm = llm.with_structured_output(MyRouteQuery)

        system = """You are an expert in directing user questions to the appropriate data source.
        Depending on the topic the question pertains to, select 2 relevant data sources.
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
        world_class — Мобильное приложение World Class 3.0, новые функции, управление расписанием и клубными услугами."""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        self.model = prompt | structured_llm

    def route_query(self, state: State):
        result = self.model.invoke({"question": state["question"]})

        res = [result.datasource_1, result.datasource_2]

        return {"context_source": res}
