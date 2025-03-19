# Чат с LLM

Этот репозиторий содержит простой чат-бот, созданный с использованием языковой модели (LLM). Бот способен отвечать на вопросы и поддерживать базовую беседу, а также сохраняет историю диалога в текущей сессии.

---

![img.png](src/img.png)

---

## Технологии

- **UI**: Для создания пользовательского интерфейса используется [Streamlit](https://streamlit.io/).
- **LLM**: В качестве языковой модели используется предоставленная API `r_m_r` с моделью `llama-3-8b-instruct-8k`.
- **Интеграция**: Для взаимодействия с LLM используется библиотека [LangChain](https://www.langchain.com/).

---

## Использование приложения
### Предварительные условия

Убедитесь что у Вас установлен и запущен Docker. 

### Запуск

Для запуска склонируйте себе данный репозиторий.

Войдите в папку app через консоль командой 
~~~ bash
cd app
~~~

Соберите контейрнер командой 
~~~ bash
docker build -t app ./
~~~

Запустите контейнер командой 
~~~ bash
docker run -p 8501:8501 app
~~~

В браузере зайдите по ссылке:
~~~
http://localhost:8501
~~~

---
# Примечания 

В файле app/.env хранятся параметры для взаимодействия с API
В нем необходимо заполнить креды для взаимодействия с API