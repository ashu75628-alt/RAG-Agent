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

    doc_prompt = PromptTemplate.from_template("""You are a friendly, casual assistant talking to a returning user.
Use the conversation history (if any) to understand follow-up questions.
Use the context below to answer naturally and warmly.
If the context does NOT contain enough information, respond with EXACTLY: "NEED_WEB_SEARCH"

Conversation History:
{history}

Context:
{context}

Question: {question}

Answer:""")

    web_prompt = PromptTemplate.from_template("""You are a friendly, casual assistant.
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
        for msg in chat_history[-6:]:  # Last 3 exchanges
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

        used_web = False
        if "NEED_WEB_SEARCH" in answer:
            used_web = True
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