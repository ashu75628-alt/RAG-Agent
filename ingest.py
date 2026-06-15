from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents():
    all_documents = []

    # Load TXT files
    print("📄 Loading TXT files...")
    try:
        txt_loader = DirectoryLoader(
            "data/",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        txt_docs = txt_loader.load()
        all_documents.extend(txt_docs)
        print(f"✅ TXT files loaded: {len(txt_docs)}")
    except Exception as e:
        print(f"No TXT files found: {e}")

    # Load PDF files
    print("📕 Loading PDF files...")
    try:
        pdf_loader = DirectoryLoader(
            "data/",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        all_documents.extend(pdf_docs)
        print(f"✅ PDF files loaded: {len(pdf_docs)}")
    except Exception as e:
        print(f"No PDF files found: {e}")

    print(f"\n📚 Total documents loaded: {len(all_documents)}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_documents)
    print(f"🔪 Total chunks created: {len(chunks)}")

    # Embed and save
    print("⏳ Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore/")
    print("✅ Vector store saved successfully!")

if __name__ == "__main__":
    ingest_documents()