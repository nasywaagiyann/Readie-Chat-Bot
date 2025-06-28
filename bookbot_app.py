import os
import pandas as pd
import streamlit as st
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# ===================== CONFIG & HEADER =====================
st.set_page_config(page_title="📚 Readie", page_icon="📘", layout="wide")

# ✅ Tampilkan Judul dan Deskripsi App SELALU
st.markdown("<h1 style='color:#2c3e50;'>🤖 Readie – Si Pintar Soal Buku</h1>", unsafe_allow_html=True)
st.markdown("📖 Mau cari bacaan baru? 🎯 Penasaran isi novel klasik? 🎨 Atau butuh rekomendasi genre? Yuk ngobrol bareng Readie di sini!")

with st.expander("💡 Contoh Pertanyaan", expanded=False):
    st.markdown("- Rekomendasi buku genre thriller")
    st.markdown("- Apa isi cerita *Animal Farm*?")
    st.markdown("- Buku serupa dengan *Harry Potter*")
    st.markdown("- Buku karya *Jane Austen*")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("🔐 Masukkan API Key Gemini")
    GOOGLE_API_KEY = st.text_input("API Key:", type="password")

    st.markdown("---")
    st.title("📁 Upload Dataset Buku (CSV)")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    st.markdown("---")
    st.caption("💡 Dapatkan API key di [Google AI Studio](https://makersuite.google.com/app/apikey)")

# ===================== VALIDASI INPUT =====================
if not GOOGLE_API_KEY:
    st.warning("🔑 Silakan masukkan API key untuk memulai.")
    st.stop()

if uploaded_file is None:
    st.warning("📂 Silakan upload file CSV berisi data buku.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ===================== LOAD DATASET =====================
@st.cache_data
def load_data_from_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    except Exception as e:
        st.error(f"❗ Gagal membaca file CSV: {e}")
        return pd.DataFrame()

    # Siapkan kolom penting
    if 'cleaned_Desc' not in df.columns:
        df['cleaned_Desc'] = ""
    else:
        df['cleaned_Desc'] = df['cleaned_Desc'].fillna("")

    return df

df = load_data_from_file(uploaded_file)

required_columns = ['Book', 'cleaned_Desc', 'Genre1', 'Genre2', 'Genre3', 'Author', 'Description']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"❗ Dataset tidak valid. Kolom berikut hilang: {', '.join(missing_cols)}")
    st.stop()

# ===================== TF-IDF =====================
vectorizer = TfidfVectorizer(stop_words="english")
try:
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_Desc"])
except ValueError:
    st.error("❗ Kolom 'cleaned_Desc' kosong atau hanya berisi stop words.")
    st.stop()

# ===================== TOOLS =====================
def get_books_by_genre(genre: str, top_n: int = 5) -> List[str]:
    genre = genre.lower()
    filtered = df[
        df['Genre1'].str.lower().fillna('').str.contains(genre) |
        df['Genre2'].str.lower().fillna('').str.contains(genre) |
        df['Genre3'].str.lower().fillna('').str.contains(genre)
    ]
    return filtered['Book'].dropna().head(top_n).tolist()

def get_description(book_title: str) -> str:
    result = df[df['Book'].str.lower() == book_title.lower()]
    return result.iloc[0]['Description'] if not result.empty else "📕 Buku tidak ditemukan."

def recommend_similar_books(book_title: str, top_n: int = 5) -> List[str]:
    index = df[df['Book'].str.lower() == book_title.lower()].index
    if len(index) == 0:
        return ["📕 Buku tidak ditemukan."]
    idx = index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    return df.iloc[similar_indices]['Book'].tolist()

def search_books_by_keywords(keywords: str, top_n: int = 5) -> List[str]:
    filtered = df[df['cleaned_Desc'].str.contains(keywords, case=False, na=False)]
    return filtered['Book'].head(top_n).tolist()

def get_author_books(author: str, top_n: int = 5) -> List[str]:
    filtered = df[df['Author'].str.lower().str.contains(author.lower())]
    return filtered['Book'].head(top_n).tolist()

# ===================== RAG + AGENT =====================
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_texts(df['cleaned_Desc'].tolist(), embedding=embedding)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    retriever=retriever,
    return_source_documents=True
)

def ask_with_rag(question: str) -> str:
    result = qa_chain.invoke({"query": question})
    return result["result"]

tools = [
    Tool(name="GetBooksByGenre", func=lambda x: str(get_books_by_genre(x)), description="List books from genre."),
    Tool(name="GetBookDescription", func=get_description, description="Get description of a book."),
    Tool(name="RecommendSimilarBooks", func=lambda x: str(recommend_similar_books(x)), description="Suggest similar books."),
    Tool(name="SearchBooksByKeywords", func=lambda x: str(search_books_by_keywords(x)), description="Search books by keyword."),
    Tool(name="GetAuthorBooks", func=lambda x: str(get_author_books(x)), description="List books by author."),
    Tool(name="AskBookContentWithRAG", func=ask_with_rag, description="Ask book content using RAG."),
]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False
)

# ===================== CHAT SECTION =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Tanyakan sesuatu tentang buku...")
if user_input:
    with st.spinner("🔄 Sedang diproses..."):
        try:
            response = agent.run(user_input)
        except Exception as e:
            response = f"❌ Error: {str(e)}"
    st.session_state.chat_history.append(("🧑 Kamu", user_input))
    st.session_state.chat_history.append(("🤖 Readie", response))

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
