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
        self.context_data = pd.read_csv(os.path.join(self.artifacts_path, dataframe_path))

        self.embedding_model = SentenceTransformer(embedding_model_name)

    def retrieve(self, state: State):
        if len(state['context_source']) == 1:
            if state['context_source'] is None or state['context_source'] == "chitchat":
                return {"context": ""}

            context = self.similarity_search(state["question"], state['context_source'], 10)

            return {"context": context['answer'].to_list()}
        else:
            text_list = []
            for name in state['context_source']:
                if name is None or name == "chitchat":
                    continue

                context = self.similarity_search(state["question"], name, 10)
                context = context.to_list()
                text_list = text_list + context

            if len(text_list) == 0:
                return {"context": ""}
            else:
                return {"context": text_list}

    def similarity_search(self, query: str, class_name: str, top_k: int = 5):
        query_emb = self.embedding_model.encode([query])
        
        index = faiss.read_index(os.path.join(self.artifacts_path, f"{class_name}.index"))
        
        _, I = index.search(query_emb, top_k)

        top_k = self.context_data[self.context_data['class_type'] == class_name].iloc[I[0]]

        return top_k


    
