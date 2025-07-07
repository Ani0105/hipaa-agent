# HIPAA Compliance Q&A Agent – Fully Groq Based (No OpenAI)

This version uses:
- SentenceTransformer for local embeddings
- FAISS for vector search
- Groq Mixtral/Gemma as the chat model

## Setup
1. Install dependencies:
   pip install -r requirements.txt

2. Add your Groq API key:
   Copy `.env.example` to `.env` and fill in:
   GROQ_API_KEY=your_key_here

3. Add your HIPAA PDFs or .txt files to the `data/` folder
   (The app builds vector DB on-the-fly inside memory.)


4. Run chatbot:
   streamlit run streamlit_app.py
