from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Проверяем наличие в базе документов
def is_db_empty():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    return vectorstore._collection.count() == 0

def split_embedding_documents(documents, on_step=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 60
    )
    chunks = splitter.split_documents(documents)
    if on_step:
        on_step(f"Разбито на {len(chunks)} чанков")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings   
        )
    vectorstore.add_documents(chunks)
    if on_step:
        on_step(f"Записано в базу")

# Индексация отдельного документа
def index_document(file_path, on_step=None):
    loader = Docx2txtLoader(file_path)
    document = loader.load()
    split_embedding_documents(document, on_step)
    
# Индексация всех файлов из папки \data
def index_all_documents(on_step=None):
    loader = DirectoryLoader(
        "data/",
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
        )
    documents = loader.load()
    split_embedding_documents(documents, on_step)