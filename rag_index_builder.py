import fitz
import os 
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf_path: str):
    """Extract text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """

    # Open the PDF file using fitz and extract text from each page
    doc = fitz.open(pdf_path)
    text = ""

    # Loop through each page in the PDF and extract text
    for page in doc:
        text += page.get_text()
    return text


def build_index_from_pdf(pdf_path: str, persistent_directory: str = "./rag_faiss_index"):
    """
    Build an index from a PDF file and store it in a persistent directory.

    Args:
        pdf_path (str): The path to the PDF file.
        persistent_directory (str): The directory where the index will be stored.
    """

    # Extract text from the PDF
    full_text = extract_text_from_pdf(pdf_path)

    # Remove the header and page numbers from the text
    full_text = re.sub(r"FIN-AP-001 \| ACCOUNTS PAYABLE POLICY \| v1\.1\s+Page \d+\s*", "", full_text)
    print(f"Extracted {len(full_text)} characters")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    # Create Document objects from the chunks
    Documents = text_splitter.split_documents([Document(page_content=full_text)])

    # Print the number of chunks created and the first 300 characters of the first chunk
    print(f"Created {len(Documents)} chunks")
    print(Documents[0].page_content[:300])

    # Create embeddings using HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_documents(Documents, embeddings)
    os.makedirs(persistent_directory, exist_ok=True)
    db.save_local(persistent_directory)
    print(f"Saved index to {persistent_directory}")


if __name__ == "__main__":
   build_index_from_pdf("./docs/Accounts_Payable_Policy.pdf", "./rag_faiss_index")