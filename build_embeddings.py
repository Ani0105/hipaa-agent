# --- Prefer a modern sqlite for Chroma on hosts like Streamlit Cloud ----------
# Must run BEFORE importing anything that may import `chromadb`
try:
    import sys, pysqlite3  # from pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
except Exception:
    pass
# -----------------------------------------------------------------------------

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# HF cache (helps on first download/build)
HF_CACHE = os.path.join(os.getcwd(), ".hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE)
os.makedirs(HF_CACHE, exist_ok=True)

def load_documents(data_dir: str = "data"):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data folder not found: {data_dir}")
    docs = []
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
    # Try Chroma; if not available (e.g. old sqlite), fall back to FAISS
    use_chroma = True
    chroma_err = None
    try:
        from langchain_community.vectorstores import Chroma
        import chromadb  # triggers sqlite version check inside the package
    except Exception as e:
        use_chroma = False
        chroma_err = e

    docs = load_documents("data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if use_chroma:
        from langchain_community.vectorstores import Chroma
        vs = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
        vs.persist()
        print("✅ Chroma DB created at ./chroma_db")
    else:
        from langchain_community.vectorstores import FAISS
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local("vector_db")
        print(f"⚠️ Chroma unavailable ({chroma_err}). Built FAISS DB at ./vector_db")

if __name__ == "__main__":
    main()
