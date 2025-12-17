import chromadb
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
COLLECTION_NAME = "csv_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-r1:14b"

st.set_page_config(page_title="CSV RAG QA", layout="wide")
st.title("📄 Question Answering using RAG with User Uploaded CSV")

# ---------------- EMBEDDING WRAPPER ----------------
class SentenceTransformerEmbedding:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# ---------------- CHROMA SINGLETON ----------------
def initialize_chroma_client():
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
    return st.session_state.chroma_client

# ---------------- LOAD CSV & CREATE VECTOR STORE ----------------
def load_data_to_vector_store(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Auto-detect first text column
    text_column = df.select_dtypes(include=["object"]).columns[0]
    texts = df[text_column].astype(str).tolist()

    documents = [
        Document(page_content=text, metadata={"row": i})
        for i, text in enumerate(texts)
        if text.strip()
    ]

    embedder = SentenceTransformerEmbedding(
        SentenceTransformer(EMBEDDING_MODEL)
    )

    client = initialize_chroma_client()

    # Remove old collection if exists (important)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
    )

    vector_store.add_documents(documents)
    return vector_store, df, text_column

# ---------------- RAG QA ----------------
def answer_query_with_rag(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model=LLM_MODEL)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
    )

    return qa.run(query)

# ---------------- STREAMLIT UI ----------------
tab1, tab2 = st.tabs(["📤 Upload CSV", "💬 Question Answering"])

with tab1:
    st.subheader("Upload CSV to Vector Database")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        with st.spinner("Indexing CSV into vector database..."):
            vector_store, df, text_column = load_data_to_vector_store(uploaded_file)
            st.session_state.vector_store = vector_store
            st.success("✅ CSV successfully indexed!")
            st.caption(f"Detected text column: `{text_column}`")
            st.dataframe(df.head())

with tab2:
    st.subheader("Ask Questions from the CSV using RAG")

    if "vector_store" not in st.session_state:
        st.warning("⚠️ Please upload a CSV file first.")
    else:
        query = st.chat_input("Ask a question about the uploaded CSV")

        if query:
            with st.spinner("Generating answer..."):
                answer = answer_query_with_rag(
                    st.session_state.vector_store,
                    query
                )

            st.chat_message("user").write(query)
            st.chat_message("assistant").write(answer)
