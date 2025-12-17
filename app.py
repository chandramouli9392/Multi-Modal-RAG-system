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

st.title("Question Answering using RAG with User Uploaded CSV")

# ---------------- EMBEDDINGS ----------------
class SentenceTransformerEmbedding:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# ---------------- CHROMA (NO PERSISTENCE) ----------------
def initialize_chroma_client():
    return chromadb.Client(
        Settings(anonymized_telemetry=False)
    )

# ---------------- LOAD CSV ----------------
def load_data_to_vector_store(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Auto-detect text column
    text_column = df.select_dtypes(include=["object"]).columns[0]
    documents_text = df[text_column].astype(str).tolist()

    documents = [
        Document(page_content=text, metadata={"row": i})
        for i, text in enumerate(documents_text)
        if text.strip()
    ]

    embedder = SentenceTransformerEmbedding(
        SentenceTransformer(EMBEDDING_MODEL)
    )

    client = initialize_chroma_client()

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
    )

    vector_store.add_documents(documents)

    return vector_store, df

# ---------------- RAG QA ----------------
def answer_query_with_rag(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model=LLM_MODEL)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    return qa.run(query)

# ---------------- STREAMLIT UI ----------------
tab1, tab2 = st.tabs(["Upload CSV", "Question Answering"])

with tab1:
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        with st.spinner("Indexing CSV..."):
            st.session_state.vector_store, df = load_data_to_vector_store(uploaded_file)
            st.success("CSV indexed successfully!")
            st.dataframe(df.head())

with tab2:
    if "vector_store" not in st.session_state:
        st.warning("Please upload a CSV first.")
    else:
        query = st.chat_input("Ask a question")

        if query:
            with st.spinner("Generating answer..."):
                answer = answer_query_with_rag(
                    st.session_state.vector_store,
                    query
                )
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(answer)
