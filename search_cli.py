from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

llm = Ollama(model="qwen2.5:7b")

question = input("Введи вопрос: ")

results = vectorstore.similarity_search(question, k=5)

context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""Ответь на вопрос, используя только этот контекст:
{context}
Вопрос: {question}
"""
answer = llm.invoke(prompt)
print(answer)