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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt for document-based answers
    doc_prompt = PromptTemplate.from_template("""You are a friendly, casual assistant.
Use the context below to answer naturally and warmly.
If the context does NOT contain enough information to answer the question, respond with EXACTLY this phrase and nothing else: "NEED_WEB_SEARCH"

Context:
{context}

Question: {question}

Answer:""")

    # Prompt for web-based answers
    web_prompt = PromptTemplate.from_template("""You are a friendly, casual assistant who talks like a helpful friend.
Use the web search results below to answer the question naturally and warmly.

Web Search Results:
{web_results}

Question: {question}

Answer (casual & friendly tone):""")

    def web_search(query):
        try:
            results = DDGS().text(query, max_results=3)
            combined = "\n\n".join([r["body"] for r in results])
            return combined
        except Exception as e:
            return "Web search failed."

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def adaptive_chain(question):
        # Step 1: Try document-based answer first
        docs = retriever.invoke(question)
        context = format_docs(docs)

        doc_chain = doc_prompt | llm | StrOutputParser()
        answer = doc_chain.invoke({"context": context, "question": question})

        # Step 2: If not enough info, fall back to web search
        if "NEED_WEB_SEARCH" in answer:
            web_results = web_search(question)
            web_chain = web_prompt | llm | StrOutputParser()
            answer = web_chain.invoke({"web_results": web_results, "question": question})
            answer = "🌐 *(Searched the web for this)*\n\n" + answer

        return answer

    class AdaptiveChain:
        def invoke(self, question):
            return adaptive_chain(question)

    return AdaptiveChain()