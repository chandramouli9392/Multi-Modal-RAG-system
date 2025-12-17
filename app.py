import chromadb
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

import warnings
warnings.filterwarnings('ignore')

# Constants
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "tweet_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-r1:14b"

# Streamlit UI setup
st.title("Question Answering using Retrival Augmented Generation with user uploaded Twitter CSV")

class SentenceTransformerEmbedding:
    """
    A wrapper class for using Sentence Transformers embeddings.
    """
    def __init__(self, model):
        self.model = model

    def embed_documents(self, documents):
        """
        Embed a list of documents (batch embedding).
        """
        return self.model.encode(documents, convert_to_tensor=True).tolist()

    def embed_query(self, query):
        """
        Embed a single query.
        """
        return self.model.encode(query, convert_to_tensor=True).tolist()

def initialize_chroma_client() -> chromadb.Client:
    """
    Initializes the Chroma client with persistent storage directory and tenant name.
    """
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIRECTORY))
    return client

def load_data_to_vector_store(uploaded_file, client) -> Chroma:
    """
    Loads CSV data into the Chroma vector store.

    Args:
        uploaded_file: The uploaded CSV file.
        client: The Chroma client instance.

    Returns:
        vector_store: The Chroma vector store instance.
    """
    df = pd.read_csv(uploaded_file)
    documents = df['body'].tolist()
    metadatas = df[['hashtags', 'author', 'mentions']].to_dict(orient='records')
    ids = df['id'].astype(str).tolist()

    # Convert documents to langchain Document objects
    documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]

    # Initialize the embedding model
    embedder = SentenceTransformerEmbedding(SentenceTransformer(EMBEDDING_MODEL))

    # Create the Chroma vector store with persistent storage
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
    )

    # Add documents to the vector store
    vector_store.add_documents(documents)
    st.success("CSV successfully stored in the vector database!")
    return vector_store, df

def answer_query_with_rag(vector_store: Chroma, query: str) -> str:
    """
    Generates an answer for the provided query using Retrieval Augmented Generation (RAG).

    Args:
        vector_store: The Chroma vector store instance.
        query: The question to be answered.

    Returns:
        answer: The generated answer.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize the Deepseek LLM
    llm = Ollama(model=LLM_MODEL)

    # Define the prompt for the question answering chain
    prompt = """
        Only use the context below.
        If unsure, say "I don’t know".
    
        Context: {context}
    
        Question: {question}
    
        Answer:
        """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    
    document_chain = create_stuff_documents_chain(llm=llm, prompt=QA_CHAIN_PROMPT)

    # Use the RetrievalQA chain for RAG
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff", 
        )

    answer = qa.run(query)
    return answer

def render_chat_interface():
    """
    Renders the interactive chat interface for Q&A.
    """
    if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []

     # Display the conversation
    for speaker, message in st.session_state['conversation_history']:
        if speaker == "User":
            # User's message in left-aligned chat bubble
            st.chat_message("user").markdown(f"**User:** <br> <div style='padding: 10px; background-color: #E0E0E0; border-radius: 10px; max-width: 80%; word-wrap: break-word; float: left;'>{message}</div>", unsafe_allow_html=True)
        else:
            # AI's message in right-aligned chat bubble
            st.chat_message("assistant").markdown(f"**AI:** <br> <div style='padding: 10px; background-color: #A5D6A7; border-radius: 10px; max-width: 80%; wrap: break-word; text-align: right; float: right;'>{message}</div>", unsafe_allow_html=True)

    # Input text box for questions at the bottom of the page
    query = st.chat_input("Ask a question")

    if query:
        # Show loading spinner while processing the answer
        with st.spinner('Generating the answer... Please wait.'):
            answer = answer_query_with_rag(vector_store, query)

        # Store the question and answer in the session state to keep track of the conversation
            st.session_state['conversation_history'].append(("User", query))
            st.session_state['conversation_history'].append(("AI", answer))

            # Display the new conversation
            st.chat_message("user").markdown(f"**User:** <br> <div style='padding: 10px; background-color: #E0E0E0; border-radius: 10px; max-width: 80%; word-wrap: break-word; float: left;'>{query}</div>", unsafe_allow_html=True)
            st.chat_message("assistant").markdown(f"**AI:** <br> <div style='padding: 10px; background-color: #A5D6A7; border-radius: 10px; max-width: 80%; wrap: break-word; text-align: right; float: right;'>{answer}</div>", unsafe_allow_html=True)

# Streamlit interface with two tabs
tab1, tab2 = st.tabs(["Upload CSV to Vector DB", "Question Answering"])

# Tab 1: Upload CSV to Chroma Vector Store
with tab1:
    st.header("Upload CSV to Vector Database")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        with st.spinner("Processing the uploaded file... Please wait."):
            client = initialize_chroma_client()
            vector_store, df = load_data_to_vector_store(uploaded_file, client)
            st.write("CSV data preview:")
            st.dataframe(df.head())

# Tab 2: Interactive Q&A using RAG
with tab2:
    st.header("Ask Questions about the data within the CSV")

    if 'vector_store' in locals():
        render_chat_interface()

