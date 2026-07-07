"""
FinSolve - Secure AI Assistant with Role-Based Access Control
A RAG chatbot where each department sees ONLY their authorized data.
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.auth import authenticate, ROLES, DEMO_USERS
from src.document_loader import (
    load_documents, chunk_documents, build_vectorstore, load_vectorstore
)
from src.rag_chain import ask_question


# ---------- Page Config ----------
st.set_page_config(
    page_title="FinSolve - Secure AI Assistant",
    page_icon="🔐",
    layout="wide"
)

# ---------- CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp { 
        background-color: #f8fafc; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Header & Branding */
    h1 { 
        font-weight: 800 !important; 
        color: #1e293b !important; 
        letter-spacing: -0.025em;
    }
    h2, h3 { 
        color: #334155 !important; 
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Login Portal Styling */
    .login-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        margin: 40px auto;
        max-width: 450px;
    }

    /* Role Badges */
    .role-badge {
        background: #eef2ff;
        color: #4338ca;
        padding: 8px 16px;
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    /* Source Badges */
    .source-badge {
        background: #f0fdf4;
        color: #166534;
        padding: 4px 12px;
        border: 1px solid #dcfce7;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Access Display Cards */
    .access-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .access-card:hover {
        border-color: #4f46e5;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.05);
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #4f46e5;
    }
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        margin-top: 4px;
    }

    /* Input & Button Styling */
    .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------- Session State ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.chat_history = []
    st.session_state.vectorstore_ready = False


# ---------- Vector Store Setup ----------
DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "data", "vectorstore")


@st.cache_resource
def setup_vectorstore():
    """Build or load the vector store."""
    if os.path.exists(os.path.join(VECTORSTORE_DIR, "chroma.sqlite3")):
        return load_vectorstore(VECTORSTORE_DIR)

    # Build from documents
    docs = load_documents(DOCS_DIR)
    if not docs:
        return None

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    vectorstore = build_vectorstore(chunks, VECTORSTORE_DIR)
    return vectorstore


# ============================================
# LOGIN PAGE
# ============================================
if not st.session_state.authenticated:
    st.markdown("# 🔐 FinSolve")
    st.markdown("*Secure AI Assistant for NexaFin Technologies*")
    st.markdown("")

    # Login form
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("### Sign In")

        username = st.text_input("Username", placeholder="e.g., sarah_cfo")
        password = st.text_input("Password", type="password", placeholder="demo123")

        if st.button("Sign In", type="primary", use_container_width=True):
            user = authenticate(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.error("Invalid credentials. Try one of the demo accounts below.")

        st.markdown("---")
        st.markdown("### Demo Accounts")
        st.markdown("*All passwords: `demo123`*")

        for uname, info in DEMO_USERS.items():
            role = info["role"]
            depts = ", ".join(ROLES[role]["departments"])
            st.markdown(
                f'<div class="access-card">'
                f'<strong>{uname}</strong> — {info["name"]}<br>'
                f'<small style="color: #a1a1aa;">Access: {depts}</small>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.stop()


# ============================================
# MAIN CHAT INTERFACE (Authenticated)
# ============================================
user = st.session_state.user

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"### Welcome, {user['name'].split('(')[0].strip()}")
    st.markdown(
        f'<span class="role-badge">{ROLES[user["role"]]["name"]}</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # API Key
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.markdown("---")

    # Access info
    st.markdown("### Your Access")
    for dept in user["departments"]:
        st.markdown(f"- {dept.title()}")

    st.markdown("---")

    # Sample questions based on role
    st.markdown("### Try asking:")
    role = user["role"]
    if role == "finance":
        samples = [
            "What was our Q1 2026 revenue?",
            "What is our investment policy?",
            "What are the office guidelines?"
        ]
    elif role == "hr":
        samples = [
            "What is the PTO policy?",
            "What's in the employee handbook?",
            "What are company values?"
        ]
    elif role == "marketing":
        samples = [
            "What's our marketing strategy for 2026?",
            "What are the social media guidelines?",
            "What are the office rules?"
        ]
    elif role == "engineering":
        samples = [
            "What is our tech stack?",
            "What are the security protocols?",
            "What are the office guidelines?"
        ]
    elif role == "executive":
        samples = [
            "What were the board meeting decisions?",
            "What is the company 3-year strategy?",
            "What was the Q1 budget allocation?"
        ]
    else:  # employee
        samples = [
            "What are the company values?",
            "What are the office guidelines?",
            "What is the dress code?"
        ]

    for s in samples:
        st.markdown(f"*\"{s}\"*")

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color: #71717a; font-size: 0.8rem;'>"
        "Built by Amar Ismail<br>"
        "<a href='https://github.com/amar-ai-engineer'>GitHub</a> | "
        "<a href='https://linkedin.com/in/amar-ai-engineer'>LinkedIn</a>"
        "</p>",
        unsafe_allow_html=True
    )


# ---------- Header ----------
st.markdown("# 🔐 FinSolve")
st.markdown(
    f"*Secure AI Assistant — Logged in as **{user['name']}** | "
    f"Access: {', '.join(user['departments'])}*"
)

# ---------- Load Vector Store ----------
with st.spinner("Loading knowledge base..."):
    vectorstore = setup_vectorstore()

if vectorstore is None:
    st.error("No documents found. Please add .txt files to data/documents/")
    st.stop()

# ---------- Chat History Display ----------
for msg in st.session_state.chat_history:
    if msg["role"] == "human":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if msg.get("sources"):
                sources_html = " ".join(
                    f'<span class="source-badge">{s}</span>'
                    for s in msg["sources"]
                )
                st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)

# ---------- Chat Input ----------
if prompt := st.chat_input("Ask about company documents..."):
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.chat_history.append({
        "role": "human",
        "content": prompt,
    })

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = ask_question(
                query=prompt,
                role=user["role"],
                user_name=user["name"],
                vectorstore=vectorstore,
                chat_history=st.session_state.chat_history,
            )

        st.write(result["answer"])

        if result["sources"]:
            sources_html = " ".join(
                f'<span class="source-badge">{s}</span>'
                for s in result["sources"]
            )
            st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)
            st.caption(f"Retrieved {result['chunks_retrieved']} relevant chunks")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
