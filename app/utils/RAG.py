from sentence_transformers import SentenceTransformer
from semantic_router.routers import SemanticRouter
from semantic_router import Route

import faiss

import pandas as pd
import numpy as np
import json
import os

from langchain_openai import ChatOpenAI


class RAG:
    def __init__(self, router_config_path: str, embedding_model_name: str,
                 classes_json_info_path: str, artifacts_path: str, dataframe_path: str):
        rl = SemanticRouter.from_json(router_config_path)
        self.rl = SemanticRouter(encoder=rl.encoder, routes=rl.routes, auto_sync="local")

        self.embedding_model = SentenceTransformer(embedding_model_name)

        with open(classes_json_info_path) as f:
            self.d_classes_info = json.load(f)

        self.artifacts_path = artifacts_path

        self.df = pd.read_csv(dataframe_path)

    def routing(self, query: str):
        result = self.rl(query)
        return result.name

    def search(self, query: str, class_name: str, top_k: int = 5):
        # получение информации о классе вопросов
        info_d = self.d_classes_info[class_name]
        indexes = np.array(info_d['indexes'])

        # инициализация vectorstore
        index = faiss.read_index(os.path.join(self.artifacts_path, f"{info_d['faiss_db_name']}.index"))

        embs = self.embedding_model.encode([query])
        D, I = index.search(embs, top_k)

        result_inds = indexes[I[0]]

        result = self.df.iloc[result_inds]

        return result

    def process(self, query: str):
        class_name = self.routing(query)

        if class_name is None:
            return []

        search_reasult = self.search(query, class_name)

        return search_reasult
