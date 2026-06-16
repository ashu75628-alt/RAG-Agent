import streamlit as st
from rag_chain import load_rag_chain
import json
from datetime import datetime

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.markdown("Ask questions from your documents!")

@st.cache_resource
def get_chain():
    return load_rag_chain()

chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

# Sidebar - Chat History
with st.sidebar:
    st.header("💬 Chat History")

    if st.button("➕ New Chat"):
        if st.session_state.messages:
            chat_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
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
            if st.button("🗑️", key=f"del_{chat_name}"):
                del st.session_state.chat_sessions[chat_name]
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear All History"):
        st.session_state.chat_sessions = {}
        st.session_state.messages = []
        st.rerun()

st.divider()

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask something..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt, chat_history=st.session_state.messages)
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })