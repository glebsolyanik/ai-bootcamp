from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np

from utils.state import State


class Reranker:
    def __init__(self):
        pass

    def bm25_reranker(self, state: State):
        """
        Ранжирует список текстов по релевантности запросу с использованием BM25.

        Параметры:
        texts (List[str]): Список текстов для ранжирования
        query (str): Поисковый запрос
        top_n (int): Опционально - количество возвращаемых результатов

        Возвращает:
        List[Tuple[str, float]]: Отсортированный список кортежей (текст, оценка релевантности)
        """
        texts = state['context']
        query = state['question']
        source = state['context_source']
        # Токенизация текстов и запроса (базовый вариант)
        tokenized_texts = [doc.split() for doc in texts]
        tokenized_query = query.split()

        # Инициализация модели BM25
        bm25 = BM25Okapi(tokenized_texts)

        # Получение оценок для всех документов
        scores = bm25.get_scores(tokenized_query)

        # Сортировка текстов по убыванию релевантности
        ranked_results = sorted(zip(texts, scores, source), key=lambda x: x[1], reverse=True)

        result = [text for text, _, _ in ranked_results]
        sources = [source for _, _, source in ranked_results]
        return {"context": result[:5], "context_source": sources[0]}
