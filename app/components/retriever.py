import os
import pandas as pd

import faiss
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from outlines import models, generate
from outlines.templates import Template

from utils.state import State
from utils.prompts import retrieval_grader_instruction


class Retriever:
    def __init__(self, artifacts_path, dataframe_path, embedding_model_name) -> None:

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
        top_k = 10
        
        for name in state['context_source']:
            if name is None or name == "chitchat":
                context_source.append(name)
                continue

            context = self.similarity_search(state["question"], name, top_k)
            context = context['answer'].to_list()
            text_list = text_list + context
            context_source.append(name)

        return {"context": text_list}


    def similarity_search(self, query: str, class_name: str, top_k: int = 5):
        query_emb = self.embedding_model.encode([query])

        index = faiss.read_index(os.path.join(self.artifacts_path, f"{class_name}.index"))

        _, I = index.search(query_emb, top_k)

        top_k = self.context_data[self.context_data['domen'] == class_name].iloc[I[0]]

        return top_k


class GradeRetrievedContext(BaseModel):
    binary_score: str = Field(
        description="Контекст имеет отношение к вопросу, 'yes' или 'no'"
    )

class RetrievalGrader:
    def __init__(self, model, api_url, api_key):
        self.model = models.openai(
            model,
            base_url=api_url,
            api_key=api_key,
        )

        self.prompt = Template.from_string(
            retrieval_grader_instruction
        )

        self.generator = generate.json(self.model, GradeRetrievedContext)

    def grade_context(self, state:State):
        is_rewrite_question = False

        filtered_context = []
        for text in state["context"]:
            score = self.generator(
                self.prompt(context=text, question=state["question"]), 
                temperature=0
            )

            grade = score.binary_score
            if grade == 'yes':
                filtered_context.append(text)
            else:
                is_rewrite_question = True
                continue

        return {"context": filtered_context, "is_rewrite_question": is_rewrite_question}

        
        