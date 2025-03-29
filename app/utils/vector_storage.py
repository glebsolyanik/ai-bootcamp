import os
import json
import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

class VectorStorage:
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        self.context_data = pd.read_csv(os.path.join(self.artifacts_path, 'clear_data.csv'))

        with open(os.path.join(self.artifacts_path, 'd_classes_json.json')) as f:
            self.d_classes_info = json.load(f)

        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

    def similarity_search(self, query: str, class_name: str, top_k: int = 5):
        query_emb = self.embedding_model.encode([query])
        
        index = faiss.read_index(os.path.join(self.artifacts_path, f"{class_name}.index"))
        
        result = index.search(query_emb, top_k)

        indexes = np.array(self.d_classes_info[class_name]['indexes'])
        result_indexes = indexes[result[1]]

        return result_indexes[0]

    def get_content(self, indexes):
        return self.context_data.iloc[indexes]