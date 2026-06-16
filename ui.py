import streamlit as st
from rag_chain import load_rag_chain

st.set_page_config(
    page_title="Knowledge Companion",
    page_icon="📚",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    .stApp {
        background-color: #0F1729;
    }

    h1 {
        font-family: 'Lora', serif !important;
        color: #F7F5F0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    .stMarkdown p, .stMarkdown {
        font-family: 'Inter', sans-serif;
        color: #C9C5BC;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px;
        margin-bottom: 8px;
    }

    [data-testid="stChatMessageContent"] {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        line-height: 1.6;
    }

    [data-testid="stSidebar"] {
        background-color: #161D2E;
        border-right: 1px solid #2A3349;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Lora', serif !important;
        color: #C9A961 !important;
    }

    .stButton button {
        background-color: #1A2236;
        color: #F7F5F0;
        border: 1px solid #2A3349;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        background-color: #C9A961;
        color: #0F1729;
        border-color: #C9A961;
    }

    [data-testid="stChatInput"] {
        background-color: #161D2E;
        border-radius: 12px;
    }

    hr {
        border-color: #2A3349 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Papa Ai")
st.markdown("*Ask anything — but backchodi allowed nhi hai, Abhi new hu tame lagega .*")

@st.cache_resource
def get_chain():
    return load_rag_chain()

chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕  New conversation", use_container_width=True):
        if st.session_state.messages:
            chat_name = f"Conversation {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[chat_name] = st.session_state.messages.copy()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    for chat_name in list(st.session_state.chat_sessions.keys()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat_name, key=f"load_{chat_name}", use_container_width=True):
                st.session_state.messages = st.session_state.chat_sessions[chat_name].copy()
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{chat_name}"):
                del st.session_state.chat_sessions[chat_name]
                st.rerun()

    st.divider()
    if st.button("Clear all history", use_container_width=True):
        st.session_state.chat_sessions = {}
        st.session_state.messages = []
        st.rerun()

st.divider()

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "📖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something..."):

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant", avatar="📖"):
        with st.spinner("Reading through your documents..."):
            response = chain.invoke(prompt, chat_history=st.session_state.messages[:-1])
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })