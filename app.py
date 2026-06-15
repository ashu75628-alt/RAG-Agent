from rag_chain import load_rag_chain

def main():
    print("🤖 RAG Chatbot Ready! Type 'quit' to exit.\n")
    chain = load_rag_chain()

    while True:
        query = input("You: ").strip()
        if query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if not query:
            continue

        response = chain.invoke(query)
        print(f"\n🤖 Bot: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()