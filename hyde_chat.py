# hyde_chat.py

from controller import Controller
from settings import config
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os
from ReferenceExtractor import ReferenceExtractor

RExt = ReferenceExtractor()
DOC_DIR = config["DOC_DIR"]

def list_available_docs():
    docs = [f for f in os.listdir(DOC_DIR) if f.endswith(".docx") or f.endswith(".pickle")]
    print("\n📄 Available documents:")
    for doc in docs:
        print(f"  - {doc}")
    return docs

def generate_hypothetical_answer(llm, query):
    prompt = f"Please write a detailed hypothetical answer to the following question:\n\n{query}"
    return llm.invoke(prompt).content

def hyde_retrieve(controller, query, selected_docs):
    # Generate hypothetical answer using LLM
    hypothetical_answer = generate_hypothetical_answer(controller.llm, query)
    print(f"\n🧪 Hypothetical Answer:\n{hypothetical_answer}\n")

    # Embed the synthetic answer
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', api_key=config["API_KEY"])
    hyde_vector = embeddings.embed_query(hypothetical_answer)

    # Build vector store filter if applicable
    filter_dict = None
    if selected_docs:
        selected_docs = [os.path.join(DOC_DIR, doc) for doc in selected_docs]
        filter_dict = {"source": {"$in": selected_docs}}

    # Search the vector DB using the hypothetical answer's embedding
    docs = controller.contextDB.vector_db.similarity_search_by_vector(hyde_vector, k=5, filter=filter_dict)

    # Return results to be used in LLM context
    return docs

def main():
    controller = Controller()
    history = []

    all_docs = list_available_docs()
    selected_docs_input = input("\nEnter document names (comma-separated) to restrict search,\nor leave blank to search across all docs: ").strip()
    selected_docs = [doc.strip() for doc in selected_docs_input.split(",")] if selected_docs_input else []

    print("\n💬 Type your questions below. Press Ctrl+C to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue

            # HyDE-enhanced document retrieval
            retrieved_docs = hyde_retrieve(controller, query, selected_docs)

            # 🖨️ Print the retrieved context
            print("\n🔍 Retrieved Context:\n" + "-" * 60)
            for i, doc in enumerate(retrieved_docs):
                print(f"[{i+1}] Source: {doc.metadata.get('source', 'N/A')}")
                print(doc.page_content[:500] + "...\n")  # Show first 500 chars
            print("-" * 60)

            # Pass retrieved docs into your existing doc_chain
            history_objs = controller.convert_history(history)
            response = controller.retriever.doc_chain.invoke({"context": retrieved_docs, "input": query, "history": history_objs})

            history.append((query, response))
            print(f"\n🤖 ChatGPT: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"[❌ Error] {e}")


if __name__ == "__main__":
    main()
