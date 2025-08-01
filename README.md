# HR Assistant RAG

A Retrieval-Augmented Generation (RAG) application for answering questions about employee records using a combination of vector search and large language models (LLMs). This project leverages [LangChain](https://python.langchain.com/) and [Ollama](https://ollama.com/) to provide an interactive HR assistant.

---

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [1. Document Loading](#1-document-loading)
  - [2. Vector Store Creation](#2-vector-store-creation)
  - [3. Retrieval-Augmented QA Chain](#3-retrieval-augmented-qa-chain)
  - [4. Streamlit Interface](#4-streamlit-interface)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Customization](#customization)
- [File Descriptions](#file-descriptions)
- [Acknowledgements](#acknowledgements)

---

## Project Structure

```
app.py
employee_records.csv
rag_pipeline.py
faiss_index/
    index.faiss
    index.pkl
__pycache__/
```

- **app.py**: Streamlit app entry point.
- **employee_records.csv**: Employee data source.
- **rag_pipeline.py**: Core RAG pipeline logic.
- **faiss_index/**: Persisted vector store for fast retrieval.
- **__pycache__/**: Python cache files.

---

## How It Works

This project implements a RAG pipeline to answer questions about employee records:

### 1. Document Loading

The function [`load_documents`](rag_pipeline.py) reads the `employee_records.csv` file and converts each row into a `Document` object, where each document contains all the employee's information as a formatted string.

### 2. Vector Store Creation

The function [`build_vector_store`](rag_pipeline.py) uses [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to embed each document. These embeddings are stored in a [FAISS](https://github.com/facebookresearch/faiss) vector database for efficient similarity search. The vector store is saved in the `faiss_index/` directory.

### 3. Retrieval-Augmented QA Chain

The function [`create_qa_chain`](rag_pipeline.py) loads the vector store and sets up a retrieval-augmented question-answering chain using:
- A retriever (from the FAISS vector store)
- An LLM (Ollama running the `llama2` model)
- [LangChain's RetrievalQA](https://python.langchain.com/docs/modules/chains/popular/qa_with_sources)

When a user asks a question, the retriever finds the most relevant employee records, and the LLM generates an answer based on those records.

### 4. Streamlit Interface

The [app.py](app.py) file provides a web interface using [Streamlit](https://streamlit.io/):
- Initializes the vector store if it doesn't exist.
- Accepts user questions.
- Displays answers and chat history.

---

## Setup Instructions

1. **Install Dependencies**

   ```sh
   pip install streamlit langchain langchain-community faiss-cpu pandas sentence-transformers
   ```

2. **Install Ollama and Download Llama2 Model**

   - [Install Ollama](https://ollama.com/download)
   - Run: `ollama pull llama2`

3. **Prepare Employee Data**

   - Place your employee records in `employee_records.csv` (ensure headers and data are correct).

---

## Usage

1. **Start Ollama**

   ```sh
   ollama serve
   ```

2. **Run the Streamlit App**

   ```sh
   streamlit run app.py
   ```

3. **Ask Questions**

   - Use the web interface to ask questions about employee records (e.g., "Who is the manager of John Doe?").

---

## Customization

- **Change Embedding Model**: Modify the `model_name` in [`build_vector_store`](rag_pipeline.py) and [`load_vector_store`](rag_pipeline.py).
- **Change LLM**: Change the model name in [`create_qa_chain`](rag_pipeline.py) (e.g., `"llama2"` to another Ollama-supported model).
- **Data Format**: Update `employee_records.csv` as needed; ensure headers match your data fields.

---

## File Descriptions

- [`app.py`](app.py): Streamlit UI, initializes the pipeline, handles user interaction.
- [`rag_pipeline.py`](rag_pipeline.py): Contains:
  - [`load_documents`](rag_pipeline.py): Loads and formats CSV data.
  - [`build_vector_store`](rag_pipeline.py): Embeds documents and builds FAISS index.
  - [`load_vector_store`](rag_pipeline.py): Loads the FAISS index for retrieval.
  - [`create_qa_chain`](rag_pipeline.py): Sets up the retrieval-augmented QA chain.
- `employee_records.csv`: Your HR data source.
- `faiss_index/`: Directory for the persisted FAISS vector store.

---

## Acknowledgements

- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)

---

**Enjoy your HR Assistant RAG!**
