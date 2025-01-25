# Overview
langcontroller is run and lets you enter your question


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