from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Проверяем наличие в базе документов
def is_db_empty():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    return vectorstore._collection.count() == 0

# Индексация отдельного документа
def index_document(file_path):
    loader = Docx2txtLoader(file_path)
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 60
    )

    chunks = splitter.split_documents(document)
    print(f"Получено кусков: {len(chunks)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings   
    )
    vectorstore.add_documents(chunks)
    print("Индексация завершена.")

# Индексация всех файлов из папки \data
def index_all_documents():
    loader = DirectoryLoader(
        "data/",
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
        )
    documents = loader.load()
    print(f"Загружено файлов: {len(documents)}")

    # Разбиваем на куски
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60
    )
    chunks = splitter.split_documents(documents)

    print(f"Получено кусков: {len(chunks)}")

    # Превращаем в векторы и сохраняем
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    print("Индексация завершена.")