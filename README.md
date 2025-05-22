# Overview
This is a 5G expert system. It uses retrieval augmented generation to augment the Q&A capabilities of an LLM. 


# Before you begin
Make sure you have a `settings.yml` file in the same directory as `settings.py`.
It should have these vars:
1. `API_KEY`
2. `DOC_DIR`
3. `MODEL_NAME`
4. `NUM_EXTRA_DOCS` -> the number of additional docs to retrieve per run. NOT depth, closer to top k
5. `CHROMA_DIR` -> The directory the chromadb sqlite db will be stored
6. `SPEC_COLL_NAME`: "context" -> These should be set to these values. They represent what the collections in the chromadb database will be named. There are two collections, one to store spec content and the other to store tdoc content.
7. `TDOC_COLL_NAME`: "reason"


# Running the system
Do `python frontend.py`


# Dependencies
gradio
langchain_openai
langchain_community
chromadb
bs4
docx2txt
MarkItDown

# Things of note:
Initializing the db only happens when the resync button is hit. 
The DB is persistent so between runs you access the same database.