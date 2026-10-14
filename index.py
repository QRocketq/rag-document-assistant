from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# 1. Загружаем ВСЕ .docx файлы из папки data/
loader = DirectoryLoader(
    "data/",
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)
documents = loader.load()

print(f"Загружено файлов: {len(documents)}")

# 2. Разбиваем на куски
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=60
)
chunks = splitter.split_documents(documents)

print(f"Получено кусков: {len(chunks)}")

# 3. Превращаем в векторы и сохраняем
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print("Индексация завершена.")