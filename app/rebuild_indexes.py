import os
import json
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

# Настройки
ARTIFACTS_PATH = "./artifacts"
DATAFRAME_PATH = "clear_data.csv"
CLASSES_JSON_INFO_PATH = "d_classes_json.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Полный путь к CSV-файлу с документами
data_path = os.path.join(ARTIFACTS_PATH, DATAFRAME_PATH)
df = pd.read_csv(data_path)
df = df.reset_index(drop=True)  # Индексы от 0 до N-1

# Определяем имена столбцов:
# - Столбец с классами (категориями) документов
class_col = "class_type"
# - Столбец с текстом документа (используется для эмбеддингов)
text_col = "answer"

# Получаем уникальные классы из CSV
unique_classes = df[class_col].unique()
logging.debug(f"Уникальные классы: {unique_classes}")

# Инициализируем модель эмбеддингов
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Новый словарь для информации о классах (будет сохранён в JSON)
new_classes_info = {}

for cls in unique_classes:
    # Фильтруем DataFrame по текущему классу
    cls_df = df[df[class_col] == cls]
    # Получаем список документов (текстов) для данного класса
    docs = cls_df[text_col].tolist()
    num_docs = len(docs)
    if num_docs == 0:
        logging.debug(f"Нет документов для класса {cls}. Пропускаем.")
        continue

    logging.debug(f"Класс '{cls}': найдено {num_docs} документов.")

    # Генерируем эмбеддинги для документов с нормализацией
    embeddings = model.encode(docs, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")  # FAISS требует float32
    dim = embeddings.shape[1]

    # Создаем FAISS индекс (используем IndexFlatIP для cosine similarity, так как эмбеддинги нормализованы)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Проверяем, что в индекс добавлено нужное количество векторов
    if index.ntotal != num_docs:
        logging.warning(f"Предупреждение: в индексе для класса {cls} {index.ntotal} векторов, ожидается {num_docs}.")
    else:
        logging.debug(f"Класс '{cls}': {num_docs} документов успешно обработано.")

    # Сохраняем новый FAISS индекс для текущего класса
    index_filename = f"{cls}.index"
    index_path = os.path.join(ARTIFACTS_PATH, index_filename)
    faiss.write_index(index, index_path)
    logging.debug(f"Индекс для класса '{cls}' сохранён в {index_path}")

    # Обновляем информацию для JSON: список индексов от 0 до (num_docs - 1)
    new_classes_info[cls] = {
        "indexes": list(range(num_docs)),
        "faiss_db_name": cls
    }

# Сохраняем обновлённый JSON-файл с информацией о классах
json_path = os.path.join(ARTIFACTS_PATH, CLASSES_JSON_INFO_PATH)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(new_classes_info, f, ensure_ascii=False, indent=2)
logging.debug(f"Обновленный JSON сохранён: {json_path}")

# Вывод содержимого JSON-файла (опционально)
print("\nСодержимое d_classes_json.json:")
with open(json_path, "r", encoding="utf-8") as f:
    print(f.read())

print("Запуск rebuild_indexes.py завершён.")


