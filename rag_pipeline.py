import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import ollama

def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = '\n'.join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content=content))
    return documents

def build_vector_store(docs, persist_dir="faiss_index"):
    embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(persist_dir)

def load_vector_store(persist_dir="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_dir, embeddings)

def create_qa_chain():
    vector_db = load_vector_store()
    llm = ollama.Ollama(model="llama2")
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    ) 
    return qa_chain
