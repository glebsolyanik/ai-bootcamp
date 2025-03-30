import os
import json
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

class VectorStorage:
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        # Загружаем датасет из CSV (с исходными документами)
        self.context_data = pd.read_csv(
            os.path.join(self.artifacts_path, st.session_state['params_RAG']['DATAFRAME_PATH'])
        )
        # Загружаем информацию о классах из JSON-файла
        with open(os.path.join(self.artifacts_path, st.session_state['params_RAG']['CLASSES_JSON_INFO_PATH']), encoding='utf-8') as f:
            self.d_classes_info = json.load(f)
        # Инициализируем модель эмбеддингов (SentenceTransformer)
        self.embedding_model = SentenceTransformer(st.session_state['params_RAG']['EMBEDDING_MODEL_NAME'])

    def index_documents(self, documents, class_name: str):
        """
        Создает и сохраняет FAISS индекс для заданного набора документов.
        """
        embeddings = self.embedding_model.encode(documents)
        embedding_dim = embeddings.shape[1]
        # Создаем индекс с метрикой Inner Product (для косинусного сходства, если эмбеддинги нормированы)
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
        index_path = os.path.join(self.artifacts_path, f"{class_name}.index")
        faiss.write_index(index, index_path)
        print(f"Проиндексировано {len(documents)} документов для класса {class_name} и индекс сохранен в {index_path}")
        return index

    def similarity_search(self, query: str, class_name: str, top_k: int = 5):
        """
        Выполняет поиск похожих документов в FAISS-индексе указанного класса.
        Если индекс существует, он будет загружен. Возвращает список глобальных индексов документов,
        отсортированных по релевантности (наиболее релевантные первым).
        """
        # Кодирование запроса в эмбеддинг
        query_emb = self.embedding_model.encode([query])
        index_path = os.path.join(self.artifacts_path, f"{class_name}.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Индекс для класса {class_name} не найден по пути: {index_path}")
        # Поиск в FAISS индексе top_k наиболее похожих векторных документов
        index = faiss.read_index(index_path)
        distances, indices = index.search(query_emb, top_k)
        # Отладочные выводы (debug logs)
        print(f"[DEBUG] Запрос: {query}")
        print(f"[DEBUG] FAISS distances: {distances}")
        print(f"[DEBUG] FAISS indices (local): {indices}")
        # Преобразование локальных индексов FAISS в глобальные индексы исходного датасета
        indexes_map = np.array(self.d_classes_info[class_name]['indexes'])
        print(f"[DEBUG] Индексы из JSON для класса '{class_name}': {indexes_map}")
        result_indexes = indexes_map[indices][0]  # получаем массив глобальных индексов
        print(f"[DEBUG] Результирующие индексы (глобальные) после сопоставления: {result_indexes}")
        # Сортируем результаты по убыванию расстояния (Inner Product): наиболее релевантные первыми
        score_index_pairs = list(zip(distances[0], result_indexes))
        score_index_pairs.sort(key=lambda x: x[0], reverse=True)
        sorted_idx = [int(idx) for (score, idx) in score_index_pairs]
        print(f"[DEBUG] Отсортированные индексы: {sorted_idx}")
        # Выводим тексты найденных документов для проверки (вопрос и ответ)
        for idx in sorted_idx:
            try:
                q_text = str(self.context_data.iloc[idx]['question'])
                a_text = str(self.context_data.iloc[idx]['answer'])
                print(f"[DEBUG] Document {idx}: Q: {q_text} | A: {a_text}")
            except Exception as e:
                print(f"[DEBUG] Невозможно получить текст документа {idx}: {e}")
        # TODO: Если используется reranker, здесь можно повторно отсортировать результаты по его оценкам,
        # чтобы на первом месте был документ с максимальной релевантностью по версии reranker.
        return sorted_idx

    def get_document_by_index(self, idx: int) -> str:
        """
        Возвращает текст ответа документа по его глобальному индексу из загруженного CSV.
        Предполагается наличие столбца 'answer' в self.context_data.
        """
        try:
            return str(self.context_data.iloc[idx]['answer'])
        except Exception as e:
            print(f"Ошибка при получении документа по индексу {idx}: {e}")
            return ""

    def get_content(self, indexes: list):
        """
        Возвращает DataFrame (строки) с содержимым документов по списку глобальных индексов.
        """
        return self.context_data.iloc[indexes]



