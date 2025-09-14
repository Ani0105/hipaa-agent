import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import base64, os
import time

# --------------------------------------------
# 1. Load environment + set dark background
# --------------------------------------------
load_dotenv()
st.set_page_config(page_title="HIPAA Q&A Agent", page_icon="🛡️", layout="centered")

# Background image setup
BG_FILE = "dark_bg.png"

def set_dark_background(img_path: str):
    if not os.path.exists(img_path):
        return  # skip if image not present
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{data}") center/cover fixed no-repeat;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }}
        .glass-card {{
            background: rgba(20, 20, 20, 0.65);
            backdrop-filter: blur(12px) saturate(120%);
            -webkit-backdrop-filter: blur(12px) saturate(120%);
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
            box-sizing: border-box;
            color: #00ffff;
            font-weight: 500;
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

        .flash {{
            animation: flash-glow 1s ease-out;
        }}
        @keyframes flash-glow {{
            from {{ filter: brightness(2.5); }}
            to {{ filter: brightness(1); }}
        }}

        [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) div.stMarkdown,
        [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) p {{
            color: #ffffff !important;
        }}
        [data-testid="stChatMessage"][data-source="user"] div.stMarkdown,
        [data-testid="stChatMessage"][data-source="user"] p {{
            color: #000000 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

set_dark_background(BG_FILE)
# Make Streamlit Secrets available as env vars when on Cloud
try:
    if "GROQ_API_KEY" in st.secrets: os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GROQ_MODEL" in st.secrets: os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]
except Exception:
    pass

# Build vector DB on first run if missing (Chroma)
if not os.path.exists("chroma_db") or not os.listdir("chroma_db"):
    from build_embeddings import main as build_main
    build_main()

# --------------------------------------------
# 2. Load QA Chain
# --------------------------------------------
@st.cache_resource
def get_qa_chain():
    # Embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vs.as_retriever()

    # LLM (Groq) – reads from .env
    model_id = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    llm = ChatGroq(model=model_id, groq_api_key=os.getenv("GROQ_API_KEY"))

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

qa_chain = get_qa_chain()

# --------------------------------------------
# 3. UI Rendering
# --------------------------------------------

# Marquee strip
st.markdown("""
<div class="marquee"><span>
📘 HIPAA = Health Insurance Portability and Accountability Act 🛡️ | It protects your private health info 🧠 | Think of it as a healthcare privacy shield 🏥🔐📜 | Your records, your rights, your rules!
</span></div>
""", unsafe_allow_html=True)

# Floating Glass Card
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
                <li>Who needs to follow HIPAA rules?</li>
            </ul>
        </li>
        <li>🔐 <strong>Privacy Rules</strong>
            <ul>
                <li>What is considered private under HIPAA?</li>
                <li>Can a doctor talk about my condition with other patients around?</li>
            </ul>
        </li>
        <li>📤 <strong>Data Sharing</strong>
            <ul>
                <li>Can a hospital send my records to an insurance company?</li>
                <li>Do I have to give permission before my health info is shared?</li>
            </ul>
        </li>
        <li>👩‍⚕️ <strong>Everyday Situations</strong>
            <ul>
                <li>Can a nurse discuss my case in a hallway?</li>
                <li>Can a receptionist call out my full name in a waiting room?</li>
            </ul>
        </li>
        <li>🚨 <strong>HIPAA Violations</strong>
            <ul>
                <li>What happens if someone breaks HIPAA rules?</li>
                <li>How do I report a HIPAA violation?</li>
            </ul>
        </li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem; text-align:center; font-size:14px;'>🤖 Developed by an AI-agent enthusiast passionate about privacy, compliance, and intelligent automation.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
