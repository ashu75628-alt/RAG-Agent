import streamlit as st
from rag_chain import load_rag_chain

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions from your documents!")
st.divider()

# Load chain only once
@st.cache_resource
def get_chain():
    return load_rag_chain()

chain = get_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask something..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
            st.markdown(response)

    # Save bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })