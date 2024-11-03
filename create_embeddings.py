import os
import uuid
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import glob

# Initialize SentenceTransformer embeddings model
embeddings_model = SentenceTransformer("all-mpnet-base-v2")

# Initialize ChromaDB client and create or get the collection
# Changed this line to use PersistentClient with a storage path
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="scrum_embeddings")


# Define the directory where your PDF files are located
directory_path = (
    r"C:\Users\thenu\Downloads\scrum-chatbot-check\scrum-chatbot-check\docs"
)

# Get all PDF file paths in the directory and its subdirectories
pdf_filepaths = glob.glob(os.path.join(directory_path, "**", "*.pdf"), recursive=True)


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def process_and_store_pdf_in_chromadb(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = [embeddings_model.encode(t).tolist() for t in texts]
    metadatas = [{"source": pdf_path, "text": t} for t in texts]
    ids = [str(uuid.uuid4()) for _ in texts]

    # Add documents to ChromaDB collection
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    print(f"Processed and stored {pdf_path}.")


# Process each PDF file and store in ChromaDB
for pdf_filepath in pdf_filepaths:
    process_and_store_pdf_in_chromadb(pdf_filepath)

print("All PDFs have been processed and stored in ChromaDB.")
