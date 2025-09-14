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
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# 1) ENV + CACHES -------------------------------------------------------------
load_dotenv()

# HF model cache to reduce 429s and cold starts
HF_CACHE = os.path.join(os.getcwd(), ".hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE)
os.makedirs(HF_CACHE, exist_ok=True)

st.set_page_config(page_title="HIPAA Q&A Agent", page_icon="🛡️", layout="centered")

# Mirror Streamlit Secrets → env (for Cloud)
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GROQ_MODEL" in st.secrets:
        os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]
    if "HUGGINGFACE_HUB_TOKEN" in st.secrets:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]
except Exception:
    pass

# Build a vector DB if none exists yet (Chroma on most hosts; FAISS fallback)
if (not os.path.isdir("chroma_db") or not os.listdir("chroma_db")) and not os.path.isdir("vector_db"):
    from build_embeddings import main as build_main
    build_main()

# 2) THEME / BG (optional) ----------------------------------------------------
BG_FILE = "dark_bg.png"

def set_dark_background(img_path: str):
    if not os.path.exists(img_path):
        return
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{data}") center/cover fixed no-repeat;
            color: #f0f0f0; font-family: 'Segoe UI', sans-serif;
        }}
        .glass-card {{
            background: rgba(20, 20, 20, 0.65);
            backdrop-filter: blur(12px) saturate(120%);
            -webkit-backdrop-filter: blur(12px) saturate(120%);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 18px; padding: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,.6); margin-top: 4rem;
        }}
        h1, h2, h4 {{ color: #ffffff; }}
        .glass-subtext {{ color: #bbbbbb; margin-bottom:1.4rem; text-align:center; }}
        .suggestions ul {{ margin-left:1.4rem; }}
        .suggestions li {{ margin-bottom: 0.6rem; }}
        .marquee {{ width:100%; overflow:hidden; white-space:nowrap;
            color:#00ffff; font-weight:500; background:rgba(0,0,0,0.4);
            padding:10px 0; font-size:15px; border-bottom:1px solid #333; }}
        .marquee span {{ display:inline-block; padding-left:100%;
            animation: marquee 22s linear infinite; }}
        @keyframes marquee {{ 0% {{ transform:translate(0,0); }}
                              100% {{ transform:translate(-100%,0); }} }}
        .lds-dual-ring {{ display:inline-block; width:32px; height:32px; }}
        .lds-dual-ring:after {{ content:" "; display:block; width:32px; height:32px;
            border-radius:50%; border:4px solid #00e0ff;
            border-color:#00e0ff transparent #00e0ff transparent;
            animation: lds-dual-ring 1.2s linear infinite; }}
        .flash {{ animation: flash-glow 1s ease-out; }}
        @keyframes flash-glow {{ from {{ filter:brightness(2.5); }} to {{ filter:brightness(1); }} }}
        [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) div.stMarkdown,
        [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) p {{ color:#ffffff !important; }}
        [data-testid="stChatMessage"][data-source="user"] div.stMarkdown,
        [data-testid="stChatMessage"][data-source="user"] p {{ color:#000000 !important; }}
        </style>
    """, unsafe_allow_html=True)

set_dark_background(BG_FILE)

# 3) LOAD VECTOR STORE (Chroma preferred, FAISS fallback) ---------------------
def load_vectorstore(embeddings):
    # Prefer Chroma if folder exists AND chroma import works
    if os.path.isdir("chroma_db") and os.listdir("chroma_db"):
        try:
            from langchain_community.vectorstores import Chroma
            vs = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
            return vs, "chroma"
        except Exception:
            pass  # fall through to FAISS

    # FAISS fallback (requires faiss-cpu on non-Windows in requirements)
    if os.path.isdir("vector_db"):
        from langchain_community.vectorstores import FAISS
        vs = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
        return vs, "faiss"

    # Nothing built for some reason
    raise RuntimeError("No vector store found. Please run build_embeddings.py.")

@st.cache_resource
def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs, store = load_vectorstore(embeddings)

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vs.as_retriever(), return_source_documents=True
    )
    st.caption(f"Vector store: {store}")
    return chain

qa_chain = get_qa_chain()

# 4) UI -----------------------------------------------------------------------
st.markdown("""
<div class="marquee"><span>
📘 HIPAA = Health Insurance Portability and Accountability Act 🛡️ | It protects your private health info 🧠 | Think of it as a healthcare privacy shield 🏥🔐📜 | Your records, your rights, your rules!
</span></div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>🛡️ HIPAA Compliance Q&A Agent</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-subtext'>Hey! I'm <strong>Anii Daa’s</strong> custom-built AI agent. Ask me anything about HIPAA!</div>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_q = st.chat_input("Type your HIPAA question here…")

    if user_q:
        with st.spinner("<div class='lds-dual-ring'></div>"):
            out = qa_chain(user_q)
        st.session_state.history.append((user_q, out["result"], out["source_documents"]))
        st.markdown("<script>document.body.classList.add('flash'); setTimeout(()=>document.body.classList.remove('flash'), 1200);</script>", unsafe_allow_html=True)

    for q, a, docs in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)
        with st.expander("📑 Sources"):
            for d in docs:
                src = d.metadata.get("source", "Unknown")
                st.markdown(f"**{src}**: …{d.page_content[:280]}…")

    st.markdown("""
    <div class='suggestions'>
    <h4>💬 You can ask me:</h4>
    <ul>
        <li>🩺 <strong>General Understanding</strong>
            <ul>
                <li>What is HIPAA and why was it created?</li>
