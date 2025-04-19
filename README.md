# Overview
This is a 5G expert system. It uses retrieval augmented generation to augment the Q&A capabilities of an LLM. 


# Before you begin
Make sure you have a `settings.yml` file in the same directory as `settings.py`.
It should have these vars:
1. `API_KEY`
2. `DOC_DIR`
3. `MODEL_NAME`
4. `IS_PICKLE` -> represents whether pickle mode is on. Pickling basically caches the documents as large byte fles so as to save on compute.
5. `NUM_EXTRA_DOCS` -> the number of additional docs to retrieve per run. NOT depth, closer to top k
6. `CHROMA_DIR` -> The directory the chromadb sqlite db will be stored


# Running the system
Do `python frontend.py`


# Dependencies
gradio
langchain_openai
langchain_community
chromadb
bs4
docx2txt

# Things of note:
Initializing the db only happens when the resync button is hit. 
The DB is persistent so between runs you access the same database.