from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# Смотрим, что есть внутри
print("=== dir(vectorstore) ===")
print()
print(dir(vectorstore))

print("\n=== help(vectorstore._collection.count) ===")
help(vectorstore._collection.count)