from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import os

load_dotenv()

def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "vectorstore/", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    doc_prompt = PromptTemplate.from_template("""You are Papa AI, a friendly and casual AI chatbot.

YOUR IDENTITY (Always answer from this if asked about yourself):
- Your name is Papa AI
- You are built by Ashutosh Kumar
- You use Groq API with LLaMA 3.3 70B model to generate answers
- You use HuggingFace Embeddings (all-MiniLM-L6-v2) for document search
- You use FAISS vector database to store and search knowledge
- You use DuckDuckGo for web search when documents don't have the answer
- You are built with Python, LangChain, and Streamlit
- You do NOT use OpenAI API

IMPORTANT RULES:
1. If anyone asks about your tech stack, APIs, creator, or how you work — ALWAYS answer from YOUR IDENTITY above, NOT from documents.
2. Only reference previous conversation if "Conversation History" below actually contains messages. If it says "No previous conversation," treat this as a brand new conversation and greet naturally.
3. Give a direct, complete answer first. Do NOT end your response with a follow-up question unless the user's request is genuinely ambiguous.
4. Keep responses focused and concise unless detail is needed.
5. Use conversation history to remember what the user told you about themselves.

Use the context below to answer naturally and warmly.
If the context does NOT contain enough information to answer (and the question is NOT about your identity), respond with EXACTLY: "NEED_WEB_SEARCH"

Conversation History:
{history}

Context:
{context}

Question: {question}

Answer:""")

    web_prompt = PromptTemplate.from_template("""You are Papa AI, a friendly and casual AI chatbot.

YOUR IDENTITY (Always answer from this if asked about yourself):
- Your name is Papa AI
- You are built by Ashutosh Kumar
- You use Groq API with LLaMA 3.3 70B model to generate answers
- You use HuggingFace Embeddings for document search
- You use FAISS vector database and DuckDuckGo for web search
- You do NOT use OpenAI API

IMPORTANT RULES:
1. If anyone asks about your tech stack, APIs, or creator — ALWAYS answer from YOUR IDENTITY above.
2. Only reference previous conversation if "Conversation History" below actually contains messages.
3. Give a direct, complete answer first. Do NOT end with a follow-up question unless necessary.
4. Use conversation history to stay consistent.

Use the conversation history and web search results to answer naturally.

Conversation History:
{history}

Web Search Results:
{web_results}

Question: {question}

Answer (casual & friendly tone):""")

    def web_search(query):
        try:
            results = DDGS().text(query, max_results=3)
            return "\n\n".join([r["body"] for r in results])
        except Exception:
            return "Web search failed."

    def format_docs_with_sources(docs):
        formatted = []
        sources = []
        for i, doc in enumerate(docs):
            source_name = doc.metadata.get("source", "Unknown").split("/")[-1].split("\\")[-1]
            page = doc.metadata.get("page", None)
            tag = f"[Source {i+1}: {source_name}" + (f", page {page+1}" if page is not None else "") + "]"
            formatted.append(f"{tag}\n{doc.page_content}")
            sources.append(f"📄 {source_name}" + (f" (page {page+1})" if page is not None else ""))
        return "\n\n".join(formatted), list(set(sources))

    def format_history(chat_history):
        if not chat_history:
            return "No previous conversation."
        formatted = []
        for msg in chat_history[-20:]:  # Last 10 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def adaptive_chain(question, chat_history=None):
        history_text = format_history(chat_history)

        docs = retriever.invoke(question)
        context, sources = format_docs_with_sources(docs)

        doc_chain = doc_prompt | llm | StrOutputParser()
        answer = doc_chain.invoke({
            "history": history_text,
            "context": context,
            "question": question
        })

        if "NEED_WEB_SEARCH" in answer:
            web_results = web_search(question)
            web_chain = web_prompt | llm | StrOutputParser()
            answer = web_chain.invoke({
                "history": history_text,
                "web_results": web_results,
                "question": question
            })
            answer = "🌐 *Searched the web for this*\n\n" + answer
        else:
            if sources:
                source_list = "\n".join(sources)
                answer += f"\n\n---\n📚 **Sources:**\n{source_list}"

        return answer

    class AdaptiveChain:
        def invoke(self, question, chat_history=None):
            return adaptive_chain(question, chat_history)

    return AdaptiveChain()