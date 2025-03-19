from langchain_openai import ChatOpenAI


def send_message_llm(context_arr, API_URL, MODEL_NAME):
    model = ChatOpenAI(base_url=API_URL, model=MODEL_NAME)
    message = ""
    for m in context_arr:
        message = message + f'Role:{m["role"]} Content: {m["content"]}\n'
    if model.get_num_tokens(message) >= 32000:
        while model.get_num_tokens(message) >= 32000:
            context_arr.pop(0)
            message = ""
            for m in context_arr:
                message = message + f'Role:{m["role"]} Content: {m["content"]}\n'
    content = model.invoke(input=message).content

    return content
