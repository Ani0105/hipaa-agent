from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

def load_documents(data_dir="data"):
    docs = []
    for fname in os.listdir(data_dir):
        fp = os.path.join(data_dir, fname)
        if fname.lower().endswith((".txt", ".md")):
            loader = TextLoader(fp, encoding="utf-8")
        elif fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(fp)
        else:
            continue
        docs.extend(loader.load())
    if not docs:
        raise ValueError("No documents found in data/")
    return docs

def main():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    vectorstore.persist()
    print("✅ Chroma DB created at ./chroma_db")


if __name__ == "__main__":
    main()
