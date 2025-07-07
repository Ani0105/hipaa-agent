import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import base64, os

# ─────────────────────────────────────────────
# 1. Load .env and set dark theme background
# ─────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="HIPAA Q&A Agent", page_icon="🛡️", layout="centered")
BG_FILE = "dark_bg.png"

def set_dark_background(img_path: str):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{b64}") center/cover fixed no-repeat;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }}
    .glass-card {{
        background: rgba(20, 20, 20, 0.65);
        backdrop-filter: blur(12px) saturate(120%);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,.6);
        margin-top: 4rem;
    }}
    h1, h2, h4 {{ color: #ffffff; }}
    .glass-subtext {{ color: #bbbbbb; margin-bottom:1.4rem; text-align:center; }}
    .suggestions ul {{ margin-left:1.4rem; }}
    .suggestions li {{ margin-bottom: 0.6rem; }}
    .marquee {{
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        color: #00ffff;
        background: rgba(0,0,0,0.4);
        padding: 10px 0;
        font-size: 15px;
        border-bottom: 1px solid #333;
    }}
    .marquee span {{
        display: inline-block;
        padding-left: 100%;
        animation: marquee 22s linear infinite;
    }}
    @keyframes marquee {{
        0% {{ transform: translate(0, 0); }}
        100% {{ transform: translate(-100%, 0); }}
    }}
    .lds-dual-ring {{
        display: inline-block;
        width: 32px;
        height: 32px;
    }}
    .lds-dual-ring:after {{
        content: " ";
        display: block;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        border: 4px solid #00e0ff;
        border-color: #00e0ff transparent #00e0ff transparent;
        animation: lds-dual-ring 1.2s linear infinite;
    }}
    </style>
    """, unsafe_allow_html=True)

set_dark_background(BG_FILE)

# ─────────────────────────────────────────────
# 2. Build vector DB from /data folder (dynamic)
# ─────────────────────────────────────────────
@st.cache_resource
def build_vector_db():
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = []
    for fname in os.listdir("data"):
        path = os.path.join("data", fname)
        if fname.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif fname.lower().endswith((".txt", ".md")):
            docs.extend(TextLoader(path, encoding="utf-8").load())

    if not docs:
        raise FileNotFoundError("📂 No files in 'data/' folder. Please add HIPAA PDFs or text files.")

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# ─────────────────────────────────────────────
# 3. Create QA Chain
# ─────────────────────────────────────────────
@st.cache_resource
def get_qa_chain():
    retriever = build_vector_db().as_retriever()
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

qa_chain = get_qa_chain()

# ─────────────────────────────────────────────
# 4. UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="marquee"><span>
📘 HIPAA = Health Insurance Portability and Accountability Act 🛡️ | It protects your private health info 🧠 | Your records, your rights, your rules!
</span></div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>🛡️ HIPAA Compliance Q&A Agent</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-subtext'>Built by <strong>Anii Daa</strong> — ask anything about HIPAA compliance!</div>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_q = st.chat_input("Type your HIPAA question here…")

    if user_q:
        with st.spinner("<div class='lds-dual-ring'></div>"):
            out = qa_chain(user_q)
        st.session_state.history.append((user_q, out["result"], out["source_documents"]))

    for q, a, docs in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)
        with st.expander("📑 Sources"):
            for d in docs:
                src = d.metadata.get("source", "Unknown")
                st.markdown(f"**{src}**: …{d.page_content[:280]}…")

    st.markdown("""
    <div class='suggestions'>
    <h4>💬 Try asking:</h4>
    <ul>
        <li>🩺 What is HIPAA and why was it created?</li>
        <li>🔐 What is considered private under HIPAA?</li>
        <li>📤 Can a hospital send my records to an insurance company?</li>
        <li>👩‍⚕️ Can a receptionist call out my full name?</li>
        <li>🚨 What happens if someone breaks HIPAA rules?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem; text-align:center; font-size:14px;'>🤖 Crafted with love for clarity and compliance.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
