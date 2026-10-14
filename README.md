# RAG Document Assistant

## Description
A local RAG (Retrieval-Augmented Generation) system that answers questions based on your documents using vector search and a local LLM.

## Technologies
- Python
- LangChain
- ChromaDB
- Ollama (qwen2.5:7b, nomic-embed-text)
- Streamlit

## How to Run
1. Install Ollama
2. Download models:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Index documents:
   ```bash
   python index.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
After running `streamlit run app.py`, a browser window opens where you can ask questions about your documents. The "Reset" button clears the chat.

## Project Structure
```
RAG project/
├── data/docs/        ← .docx documents
├── db/               ← vector database
├── index.py          ← indexing script
├── app.py            ← web interface
├── search_cli.py     ← CLI version
├── requirements.txt  ← dependencies
└── README.md         ← this file
```