import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
import os

st.set_page_config(page_title="RAG PDF QA App", layout="wide")
st.title("📚 Ask Questions About Your PDF Files")

# Sidebar for uploading PDFs
uploaded_files = st.sidebar.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    with st.spinner("Processing documents..."):
        # Save uploaded files to temp files
        temp_dir = tempfile.TemporaryDirectory()
        pdf_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            pdf_paths.append(path)

        # Load and split PDFs
        all_docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        # Embed documents
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(chunks, embeddings)

        # Setup local Hugging Face model
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # RAG QA chain
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # User input
        question = st.text_input("Ask a question about your PDFs:", placeholder="e.g., What is ICT infrastructure?")

        if question:
            result = qa_chain.invoke(question)
            answer = result["result"]
            sources = result.get("source_documents", [])

            # Display answer
            st.markdown("### 🤖 Answer")
            st.write(answer)

            # Optional: show sources
            with st.expander("📄 Source Chunks"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content)

            # Save chat history
            st.session_state.chat_history.append((question, answer))

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Chat History")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
else:
    st.info("Upload PDF files to get started.")
