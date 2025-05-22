# cli_chat.py

from controller import Controller
from settings import config
import os
from ReferenceExtractor import ReferenceExtractor
from langchain_core.documents import Document

RExt = ReferenceExtractor()

def list_available_docs():
    doc_dir = config["DOC_DIR"]
    docs = [f for f in os.listdir(doc_dir) if f.endswith(".docx") or f.endswith(".pickle")]
    print("\n📄 Available documents:")
    for doc in docs:
        print(f"  - {doc}")
    return docs

def doc_id_to_filename(doc_id, available_docs):
    """Helper to match doc_id like '38.211' to actual filename in DOC_DIR."""
    for file in available_docs:
        if doc_id in file:
            return os.path.join(config["DOC_DIR"], file)
    return None

def main():
    controller = Controller()
    history = []

    use_rag = input("Use RAG (vector DB context)? [y/n]: ").strip().lower()
    if use_rag == 'n':
        controller.toggleDatabase()
        print("🔌 RAG DISABLED")
    else:
        print("🔌 RAG ENABLED")

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

            # Wrap prompt in Document for compatibility
            prompt_doc = Document(page_content=prompt)

            # Reference extraction
            doc_ids = RExt.extractDocIdsFromStrList([prompt])
            clause_nums = RExt.extractClauseNumbersOfSrc(RExt.runREWithDocList([prompt_doc]))

            print(f"📘 Detected doc IDs: {doc_ids}")
            print(f"📑 Detected clause numbers: {clause_nums}")

            # Map doc IDs to actual filenames
            extra_filtered_docs = [doc_id_to_filename(doc_id, all_docs) for doc_id in doc_ids]
            extra_filtered_docs = [doc for doc in extra_filtered_docs if doc]  # Remove None

            # Merge user-selected docs with extracted ones
            all_selected = list(set(
                [os.path.join(config["DOC_DIR"], doc) for doc in selected_docs] + extra_filtered_docs
            )) if selected_docs or doc_ids else None

            # Apply retriever
            controller.retriever.constructRetriever(db=controller.contextDB, selected_docs=all_selected)
            history_objs = controller.convert_history(history)

            if controller.isDatabaseTriggered:
                resp, orig_docs, additional_docs = controller.getResponseWithRetrieval(prompt, history_objs)

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

