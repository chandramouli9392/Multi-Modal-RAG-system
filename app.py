import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="CSV RAG QA", layout="wide")
st.title("📄 Question Answering using RAG with User Uploaded CSV")

# ---------------- EMBEDDINGS ----------------
class SentenceTransformerEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# ---------------- LOAD CSV ----------------
def load_data_to_vector_store(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Automatically detect first text column
    text_column = df.select_dtypes(include=["object"]).columns[0]
    texts = df[text_column].astype(str).tolist()

    documents = [
        Document(page_content=text, metadata={"row": i})
        for i, text in enumerate(texts)
        if text.strip()
    ]

    embeddings = SentenceTransformerEmbedding()

    vector_store = InMemoryVectorStore.from_documents(
        documents, embedding=embeddings
    )

    return vector_store, df, text_column

# ---------------- RAG PIPELINE ----------------
def answer_query_with_rag(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Use ONLY the context below to answer the question.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=st.secrets["HF_TOKEN"],
        model_kwargs={"temperature": 0.2, "max_length": 256}
    )

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    return llm.invoke(final_prompt)

# ---------------- STREAMLIT UI ----------------
tab1, tab2 = st.tabs(["📤 Upload CSV", "💬 Question Answering"])

with tab1:
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        with st.spinner("Indexing CSV into vector store..."):
            vector_store, df, text_column = load_data_to_vector_store(uploaded_file)
            st.session_state.vector_store = vector_store
            st.success("✅ CSV indexed successfully")
            st.caption(f"Detected text column: `{text_column}`")
            st.dataframe(df.head())

with tab2:
    if "vector_store" not in st.session_state:
        st.warning("⚠️ Please upload a CSV first")
    else:
        query = st.chat_input("Ask a question about the CSV data")

        if query:
            with st.spinner("Generating answer..."):
                answer = answer_query_with_rag(
                    st.session_state.vector_store,
                    query
                )

            st.chat_message("user").write(query)
            st.chat_message("assistant").write(answer)
