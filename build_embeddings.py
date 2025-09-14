# --- Fix sqlite version for Chroma on some hosts (e.g., Streamlit Cloud) ---
# Must run BEFORE importing chromadb/langchain_community.vectorstores
try:
    import sys, pysqlite3  # provided by pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
except Exception:
    pass
# ---------------------------------------------------------------------------

import os
from dotenv import load_dotenv

# LangChain loaders / embeddings / vector store
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Cache HF models on disk so cold starts don’t re-download every time
HF_CACHE = os.path.join(os.getcwd(), ".hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE)
os.makedirs(HF_CACHE, exist_ok=True)

def load_documents(data_dir: str = "data"):
    docs = []
    if not os.path.exists(data_dir):
        raise ValueError(f"Data folder not found: {data_dir}")

    for fname in os.listdir(data_dir):
        fp = os.path.join(data_dir, fname)
        if not os.path.isfile(fp):
            continue
        lower = fname.lower()
        if lower.endswith((".txt", ".md")):
            loader = TextLoader(fp, encoding="utf-8")
        elif lower.endswith(".pdf"):
            loader = PyPDFLoader(fp)
        else:
            continue
        docs.extend(loader.load())

    if not docs:
        raise ValueError("No documents found in data/")
    return docs

def main():
    docs = load_documents("data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db",
    )
    vectorstore.persist()
    print("✅ Chroma DB created at ./chroma_db")

if __name__ == "__main__":
    main()
