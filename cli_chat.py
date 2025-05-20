# cli_chat.py

from controller import Controller
from settings import config
import os

def list_available_docs():
    doc_dir = config["DOC_DIR"]
    docs = [f for f in os.listdir(doc_dir) if f.endswith(".docx") or f.endswith(".pickle")]
    print("\n📄 Available documents:")
    for doc in docs:
        print(f"  - {doc}")
    return docs

def main():
    controller = Controller()
    history = []

    # --- RAG Toggle ---
    use_rag = input("Use RAG (vector DB context)? [y/n]: ").strip().lower()
    if use_rag == 'n':
        controller.toggleDatabase()
        print("🔌 RAG DISABLED")
    else:
        print("🔌 RAG ENABLED")

    # --- Doc Selection ---
    all_docs = list_available_docs()
    selected_docs_input = input("\nEnter document names (comma-separated) to restrict search,\nor leave blank to search across all docs: ").strip()

    if selected_docs_input:
        selected_docs = [doc.strip() for doc in selected_docs_input.split(",")]
    else:
        selected_docs = []

    print("\n💬 Type your questions below. Press Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            # Reconstruct retriever if necessary (to apply selected docs)
            controller.retriever.constructRetriever(db=controller.contextDB, selected_docs=selected_docs)
            history_objs = controller.convert_history(history)

            if controller.isDatabaseTriggered:
                # Access full RAG output
                resp = controller.retriever.invoke(query=prompt, history=history_objs, db=controller.contextDB)
                print("\n🔍 Retrieved Context:\n" + "-" * 60)
                for i, doc in enumerate(resp.get("context", [])):
                    print(f"[{i+1}] {doc.page_content[:500]}...\n")
                print("-" * 60)

                answer = resp["answer"]
            else:
                chain = controller.prompt | controller.llm
                resp = chain.invoke({"input": prompt, "history": history_objs})
                answer = resp.content

            history.append((prompt, answer))
            print(f"\n🤖 ChatGPT: {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"[❌ Error] {e}")

if __name__ == "__main__":
    main()

