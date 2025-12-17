import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore

st.set_page_config(page_title="Simple CSV RAG", layout="centered")
st.title("📄 Simple RAG QA (No HF, No FAISS)")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def build_vector_store(df, text_col):
    docs = [
        Document(page_content=str(row[text_col]), metadata=row.to_dict())
        for _, row in df.iterrows()
    ]

    vector_store = InMemoryVectorStore.from_documents(
        docs,
        embedding=lambda texts: embedder.encode(texts).tolist()
    )
    return vector_store

st.header("1️⃣ Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV Loaded Successfully")
    st.dataframe(df.head())

    text_column = st.selectbox("Select text column for RAG", df.columns)

    if st.button("Build Vector Store"):
        with st.spinner("Creating embeddings..."):
            st.session_state.vector_store = build_vector_store(df, text_column)
        st.success("Vector Store Ready")

st.header("2️⃣ Ask Questions")

query = st.text_input("Ask a question based on the CSV")

if query and "vector_store" in st.session_state:
    with st.spinner("Searching relevant context..."):
        retriever = st.session_state.vector_store.as_retriever(k=3)
        docs = retriever.invoke(query)

    st.subheader("🔍 Retrieved Context")
    for i, doc in enumerate(docs, 1):
        st.markdown(f"**Chunk {i}:** {doc.page_content}")

    st.subheader("🧠 Answer")
    answer = " ".join([doc.page_content for doc in docs])
    st.write(answer)

elif query:
    st.warning("Please upload CSV and build vector store first.")
