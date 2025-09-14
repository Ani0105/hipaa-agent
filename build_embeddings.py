from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

DATA_DIR = "data"
VDB_DIR = "vector_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_documents(data_dir: str = DATA_DIR):
    docs = []
    if not os.path.isdir(data_dir):
        raise ValueError(f"Missing data folder: {data_dir}")

    for fname in os.listdir(data_dir):
        fp = os.path.join(data_dir, fname)
        name = fname.lower()
        if name.endswith((".txt", ".md")):
            docs.extend(TextLoader(fp, encoding="utf-8").load())
        elif name.endswith(".pdf"):
            docs.extend(PyPDFLoader(fp).load())
        else:
            # skip non-text/pdf assets
            continue

    if not docs:
        raise ValueError(f"No loadable .pdf/.txt/.md files found in {data_dir}/")
    return docs

def main():
    print("🔧 Building FAISS index...")
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vstore = FAISS.from_documents(chunks, embeddings)
    vstore.save_local(VDB_DIR)
    print(f"✅ FAISS DB saved at ./{VDB_DIR}")

if __name__ == "__main__":
    main()
