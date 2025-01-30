# Overview
This is a 5G expert system. It uses retrieval augmented generation to augment the Q&A capabilities of an LLM. 


# Before you begin
Make sure you have a `settings.yml` file in the same directory as `settings.py`.
It should have these vars:
1. `API_KEY`
2. `DOC_DIR`
3. `MODEL_NAME`
4. `IS_PICKLE` -> represents whether pickle mode is on. Pickling basically caches the documents as large byte fles so as to save on compute.

# Running the system
Do `python frontend.py`


# Dependencies
gradio
langchain_openai
langchain_community
chromadb
pypdf

# Goals:
Get System working again
Refactor & clean up
Build automatic fetching of docs