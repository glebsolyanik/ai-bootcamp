import os
import json
import numpy as np
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer

from utils.state import State

class Retriever:
    def __init__(self, 
            artifacts_path,
            dataframe_path,
            classes_json_info_path,
            embedding_model_name,
        ) -> None:

        self.artifacts_path = artifacts_path
        self.context_data = pd.read_csv(os.path.join(self.artifacts_path, dataframe_path))

        with open(os.path.join(self.artifacts_path, classes_json_info_path)) as f:
            self.d_classes_info = json.load(f)

        self.embedding_model = SentenceTransformer(embedding_model_name)

    def retrieve(self, state:State):
        if state['context_source'] is None or state['context_source'] == "chitchat":
            return {"context": ""}
        
        indexes = self.similarity_search(state["question"], state['context_source'])
        content = self.get_content(indexes)['answer'].to_list()

        return {"context": content}

    def similarity_search(self, query: str, class_name: str, top_k: int = 5):
        query_emb = self.embedding_model.encode([query])
        
        index = faiss.read_index(os.path.join(self.artifacts_path, f"{class_name}.index"))
        
        result = index.search(query_emb, top_k)

        indexes = np.array(self.d_classes_info[class_name]['indexes'])
        result_indexes = indexes[result[1]]

        return result_indexes[0]

    def get_content(self, indexes):
        return self.context_data.iloc[indexes]


    
