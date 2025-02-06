import os
import pypdfium2
import yaml
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def create_embeddings(embeddings_path=config["embeddings_path"]):
    return SentenceTransformerEmbeddings(model_name=embeddings_path)

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient("chroma_db")
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings,
    )
    return langchain_chroma

def get_pdf_texts(pdfs_bytes_list):
    """Extract text from a list of PDF byte streams."""
    return [extract_text_from_pdf(pdfs_bytes_list)]

def extract_text_from_pdf(pdf_bytes):
    """Extract text from a single PDF byte stream."""
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(
        pdf_file.get_page(page_number).get_textpage().get_text_range()
        for page_number in range(len(pdf_file))
    )

def get_text_chunks(text):
    """Split text into chunks of a specified size."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, separators=["\n", "\n\n"])
    return splitter.split_text(text)

def get_document_chunks(text_list):
    """Convert a list of text strings into Document chunks."""
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))
    return documents

def add_to_db(doc_path, chat_id, collection):
    metadata = []
    data = []
    ids = []
    with open(doc_path, "rb") as f:
        pdfs_bytes = f.read()
    texts = get_pdf_texts(pdfs_bytes)
    documents = get_document_chunks(texts)
    for d in documents:
        metadata.append({"chat_id": chat_id})
        ids.append(str(uuid4()))
        data.append(d.page_content)
    collection.add(documents=data, metadatas=metadata, ids=ids)

# run chroma with command: chroma run --path test --port 9000