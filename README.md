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

# Refactoring Plan
1. Selected docs really needs to be reworked. CUrrently hardcoded. needs the `\\` to work with windows.
2. Too many random print statements!
3. Lift things up into APIs and folders. Make this a module!
4. CLean up requirements
5. Add pickling as an option through argparse

# Goals:
Get System working again
Refactor & clean up
Build automatic fetching of docs