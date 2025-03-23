![Робот](src/Робот.png)

# Мульти-чат с LLM (LangChain + LangGraph + PostgreSQL)

Данный проект представляет собой веб-приложение на базе **Streamlit**, которое позволяет:
- Общаться с большой языковой моделью (LLM) в режиме чата.
- Создавать и хранить несколько чатов в базе данных **PostgreSQL**.
- Управлять «температурой» и другими параметрами модели.
- Легко настраивать подключение к локальной или удалённой LLM (например, OpenAI).

---
## Основные возможности

1. **Несколько чатов**: Создавайте новые чаты, переименовывайте и переключайтесь между ними.
2. **История сообщений**: Хранится в PostgreSQL, поэтому при перезапуске приложения контекст не теряется.
3. **Гибкая настройка модели**: Укажите поставщика, название модели, URL и API-ключ.
4. **Streamlit-интерфейс**: Удобный веб-приложение, не требующее дополнительной фронтенд-разработки.

---
# Как настроить чат

![Ключи](src/Ключи.png)

1. **Название модели** — введите LLM, с которой планируете работать (например, `llama-2-7b`).
2. **Поставщик модели** — укажите `OpenAI` (или другого провайдера).
3. **Базовый URL** — напишите URL модели, например: `https://api.openai.com/v1/chat/completions`.
4. **API ключ** — вставьте ваш секретный ключ (не передавайте его никому!).
5. Нажмите на кнопку **«Сохранить настройки»** — и можно начинать общение в чате.

---
## Модель работы в чате

![Чат](src/Чат.png)

1. В левой боковой панели выберите или создайте новый чат.
2. В главном окне введите сообщение и нажмите Enter.
3. Модель сгенерирует ответ, учитывая предыдущий контекст диалога.
4. Все сообщения сохраняются в базе данных, чтобы вы могли вернуться к ним позже.

---
## Установка и запуск

### 1. Локальный запуск (без Docker)

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/username/ai-bootcamp.git
   cd ai-bootcamp/app


2. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Настройте PostgreSQL**:
   - Запустите PostgreSQL локально или настройте подключение к удалённой базе данных.
   - Убедитесь, что переменные окружения (`DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`) настроены корректно.

4. **Запустите Streamlit**:
   ```bash
   streamlit run main.py
   ```

5. **Откройте приложение**:
   - Перейдите по адресу `http://localhost:8501` в браузере.

---

### 2. Запуск через Docker Compose

1. **Убедитесь, что у вас установлены Docker и Docker Compose**:
   - Если Docker не установлен, следуйте официальной документации: [Docker Installation](https://docs.docker.com/get-docker/).

2. **Запустите приложение**:
   - В корневой папке (где находится `docker-compose.yml`) выполните:
     ```bash
     docker-compose up -d
     ```

3. **Откройте приложение**:
   - Перейдите по адресу `http://localhost:8501` в браузере.

4. **Остановка приложения**:
   - Для остановки выполните:
     ```bash
     docker-compose down
     ```
---

## Переменные окружения

Приложение использует переменные окружения (через `.env` или иными способами):

- **Для подключения к PostgreSQL**:
  - `DB_NAME` — имя базы данных.
  - `DB_USER` — пользователь базы данных.
  - `DB_PASSWORD` — пароль пользователя.
  - `DB_HOST` — хост базы данных.
  - `DB_PORT` — порт базы данных.

- **Для настройки LLM**:
  - `MODEL_NAME` — название модели (например, `llama-2-7b`).
  - `MODEL_PROVIDER` — поставщик модели (например, `OpenAI`).
  - `API_URL` — URL API модели (например, `https://api.openai.com/v1/chat/completions`).
  - `API_KEY` — ваш API-ключ (храните его в безопасности!).

---
## Возможные проблемы и решения

1. **Не удаётся подключиться к БД**:
   - Проверьте настройки в `.env`.
   - Убедитесь, что PostgreSQL запущен и доступен.

2. **LLM не отвечает**:
   - Проверьте, что указаны правильные `API_KEY`, `API_URL` и `MODEL_PROVIDER`.
   - Убедитесь, что API-ключ активен и имеет доступ к модели.

3. **Ошибка Docker**:
   - Проверьте, что файлы `Dockerfile` и `docker-compose.yml` корректны.
   - Убедитесь, что Docker и Docker Compose установлены правильно.

4. **Порт занят**:
   - Измените порт в `docker-compose.yml` на свободный.
   - Или освободите занятый порт, завершив процесс, который его использует.

---

- Принимаются Pull Request'ы и Issue для улучшения проекта!




