import streamlit as st
from rag_pipeline import load_documents, build_vector_store, create_qa_chain
import os 

st.set_page_config(page_title=" HR Assistant RAG", page_icon=":robot_face:"
                   )
st.title("HR Assistant RAG")

# initialize the vector store if it doesn't exist
if not os.path.exists("faiss_index"):
    with st.spinner("Indexing employees data..."):
        docs = load_documents("employee_records.csv")
        build_vector_store(docs)

# Create the QA chain
qa_chain = create_qa_chain()

# chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about employyee records:")

if user_input:
    response = qa_chain.run(user_input)
    st.session_state.chat_history.append((user_input, response))

# Display chat history
for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
    st.markdown(f"**🧑‍💼 You:** {q}")
    st.markdown(f"**🤖 HR Bot:** {a}")