from openai import OpenAIError
from langchain.chat_models import init_chat_model

from utils.state import State

class LLM:
    def __init__(self, model, model_provider, api_url, api_key) -> None:
        self.model = init_chat_model(
            model=model,
            model_provider=model_provider,
            base_url=api_url,
            api_key=api_key,
        )

    def validate_model(self):
        try:
            self.model.invoke("test")
            return True
        except OpenAIError:
            return False

    def generate(self, state:State):
        if state["context"] == "":
            prompt = f"""Общайся с пользователем
            История переписки: {state["messages"]}
            Сообщение пользователя: {state["question"]}
            """
        else:
            prompt = f"""Пользователь задал вопрос.
                Используя контекст, дай ему ответ на вопрос. Ответ должен быть емким, и опираться на контекст. 
        
                История переписки: {state["messages"]}
                
                Контекст: {state["context"]}
                
                Вопрос: {state["question"]}
                
                В конце добавь приписку, без изменений: Данные взяты из: {state['context_source']}  
                """
        response = self.model.invoke(prompt)

        return {"messages": response}

    
