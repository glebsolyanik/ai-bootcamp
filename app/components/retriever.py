import os
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer

from utils.state import State


class Retriever:
    def __init__(self,
                 artifacts_path,
                 dataframe_path,
                 embedding_model_name,
                 ) -> None:

        self.artifacts_path = artifacts_path

        self.path_to_context_data = os.path.join(self.artifacts_path, dataframe_path)
        self.context_data = None


        self.embedding_model = SentenceTransformer(embedding_model_name)


    def init_context_data(self):

        self.context_data = pd.read_csv(self.path_to_context_data)

        return True
    def retrieve(self, state: State):
        text_list = []
        context_source = []
        for name in state['context_source']:
            if name is None or name == "chitchat":
                return {"context": [], "context_source": []}

            context = self.similarity_search(state["question"], name, 10)
            context = context['chunk'].to_list()
            text_list = text_list + context
            context_source = context_source + [name]

        return {"context": text_list, "context_source": context_source}


    def similarity_search(self, query: str, class_name: str, top_k: int = 5):
        query_emb = self.embedding_model.encode([query])

        index = faiss.read_index(os.path.join(self.artifacts_path, f"{class_name}.index"))

        _, I = index.search(query_emb, top_k)

        top_k = self.context_data[self.context_data['domen'] == class_name].iloc[I[0]]

        return top_k
