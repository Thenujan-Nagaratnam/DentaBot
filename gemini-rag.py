from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Constants
MODEL_TYPE = "gemini"  # Choose LLM model, Gemini by default
PERSIST_DIRECTORY = "db"
DOC_DIRECTORY = "docs"
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
PDF_GLOB_PATTERN = "./*.pdf"
TEXT_GLOB_PATTERN = "./*.txt"


def initialize_model(model_type):
    """Initialize the model based on the selected type."""
    if model_type == "gemini":
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            convert_system_message_to_human=True,
        )
    else:
        try:
            model = Ollama(
                model=model_type,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
            return model
        except Exception as e:
            print(f"Error: Could not load the model '{model_type}'. Details: {str(e)}")


def load_data():
    """Load and split PDF and text data."""
    pdf_loader = DirectoryLoader(
        DOC_DIRECTORY, glob=PDF_GLOB_PATTERN, loader_cls=PyPDFLoader
    )
    text_loader = DirectoryLoader(
        DOC_DIRECTORY, glob=TEXT_GLOB_PATTERN, loader_cls=TextLoader
    )

    pdf_documents = pdf_loader.load()
    text_documents = text_loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
    text_context = "\n\n".join(str(p.page_content) for p in text_documents)

    pdfs = splitter.split_text(pdf_context)
    texts = splitter.split_text(text_context)

    return pdfs + texts


def create_vector_db(data, embeddings, persist_directory):
    """Create and persist a vector database."""
    vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
    return vectordb


def load_or_create_vectordb(embeddings, persist_directory):
    """Load an existing vector database or create a new one."""
    if os.path.exists(persist_directory):
        print("Vector DB Loaded\n")
        return Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
    else:
        print("Persist directory does not exist. Creating new Vector DB...")
        data = load_data()
        vectordb = create_vector_db(data, embeddings, persist_directory)
        print("Vector DB Created and Persisted\n")
        return vectordb


def initialize_retrieval_chain(model, vectordb):
    """Initialize the RetrievalQA chain with a custom prompt."""

    # Create the RetrievalQA chain
    query_chain = RetrievalQA.from_chain_type(
        llm=model, retriever=vectordb.as_retriever()
    )

    # Define a custom prompt function that adds the custom instructions
    def custom_query_function(user_query):

        # Combine the last 10 messages from the conversation history
        history_context = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.history[-10:]
            ]
        )

        full_prompt = f"""Assume you are a Dental chatbot.
        Answer only queries related to it in a professional and detailed manner:

        Context from uploaded documents, use this only as an additional input to your existing knowledge, if it is related to the query or else ignore it and use your own knowledge. Prioritize your own knowledge in any case and ignore the context from uploaded documents, if your own knowledge itself has a better answer. If the query is not related to dental, say I cannot answer out of context or something similar. \n\nPrevious conversation:\n{history_context}\n\nUser query: {user_query}"""

        # Get the relevant documents based on the query
        retrieved_docs = vectordb.as_retriever().get_relevant_documents(full_prompt)
        print(retrieved_docs)

        return query_chain({"query": full_prompt})

    return custom_query_function


def chat_interface(query_chain):
    """Handle the Streamlit chat interface."""
    if "history" not in st.session_state:
        st.session_state.history = []

    # Render chat history first to avoid duplicating the latest interaction
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Say something")
    if prompt:
        # Add user's input to the session state history and display it immediately
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the assistant's response
        with st.spinner("Thinking 💡"):
            response = query_chain({"query": prompt})

            assistant_response = response["result"]
            st.session_state.history.append(
                {"role": "Assistant", "content": assistant_response}
            )
            with st.chat_message("Assistant"):
                st.markdown(assistant_response)


def main():
    """Main function to run the chatbot application."""
    # Initialize the model
    model = initialize_model(MODEL_TYPE)

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load or create the vector database
    vectordb = load_or_create_vectordb(embeddings, PERSIST_DIRECTORY)

    # Initialize the retrieval chain
    query_chain = initialize_retrieval_chain(model, vectordb)

    # Run the chat interface
    chat_interface(query_chain)


if __name__ == "__main__":
    main()
