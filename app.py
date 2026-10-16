import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from index import is_db_empty, index_all_documents

# Проверяем наличие документов в базе, если база пуста индексируем все из папки data\
if is_db_empty():
    index_all_documents()
# --- 1. Загружаем всё один раз при запуске приложения ---
# Streamlit перезапускает скрипт при каждом взаимодействии.
# Чтобы не перезагружать модель и базу каждый раз, используем кэш.
@st.cache_resource
def load_components():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    llm = Ollama(model="qwen2.5:7b")
    return vectorstore, llm

vectorstore, llm = load_components()

# --- 2. Настраиваем страницу ---
st.title("Мой RAG-помощник")
st.caption("Задайте вопрос по документам, который вы загрузили в базу.")

# --- 3. Инициализируем историю чата в памяти сессии ---
# Streamlit хранит данные между перезапусками в st.session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Создание кнопки "Reset" ---
def c_cleaner():
    st.session_state.messages = []

st.button("Reset", type="primary", on_click = c_cleaner)
# --- 5. Отображаем всю историю сообщений ---
# Каждое сообщение — это словарь с полями "role" (user/assistant) и "content".
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- 6. Реагируем на ввод пользователя ---
# st.chat_input создаёт поле ввода внизу страницы.
if prompt := st.chat_input("Введите ваш вопрос..."):
    # Показываем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 7. Логика RAG (та же, что была в search.py) ---
    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ в документах..."):
            # Поиск похожих кусков
            results = vectorstore.similarity_search(prompt, k=10)
            # Собираем контекст
            context = "\n\n".join([doc.page_content for doc in results])

            # Формируем промпт
            full_prompt = f"""Ты — помощник, который отвечает на русском языке.
Используй информацию из контекста ниже, чтобы ответить на вопрос.
Если в контексте есть related информация — используй её.
Не используй китайский язык. Можешь добавить вывод от себя на основе этой информации.

Контекст:
{context}

Вопрос: {prompt}
"""

            # Получаем ответ от модели
            answer = llm.invoke(full_prompt)

            # Показываем ответ
            st.markdown(answer)

    # Сохраняем ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": answer})