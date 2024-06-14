import gradio as gr

from controller import Controller
from langchain.chains import ConversationChain
from typing import Optional, Tuple
from threading import Lock
from settings import config
import os

langchain_controller = Controller()
langchain_controller.createVectorStore()


def respond(prompt,history,selected_docs):
    resp = langchain_controller.runController(prompt,history,selected_docs)
    history.append((prompt,resp))
    return history


def adjustToggle():
    IS_DB = langchain_controller.toggleDatabase()
    if IS_DB:
        text = "RAG ENABLED CURRENTLY"
    else:
        text = "RAG DISABLED CURRENTLY"
    return text



# idea: design function for the submit. This function will have as inputs, prompt and history
# get history from chatbot. As output, have "", history so both q and answer will go into chatbot

DOC_DIR = config["DOC_DIR"]
file_lst = []
for file in os.listdir(DOC_DIR):
    if file[-7:] == ".pickle":
        file = file[:-7]
    file_lst.append(file)

docs = file_lst
with gr.Blocks() as demo:
    toggleDB = gr.Button("RAG ENABLED CURRENTLY")
    toggleDB.click(fn=adjustToggle,outputs = toggleDB)
    chatbot=gr.Chatbot(height=500)
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7)
    selected_docs = gr.Dropdown(choices=docs, multiselect=True)
    textbox.submit(respond,inputs=[textbox,chatbot,selected_docs],outputs=chatbot)

demo.launch(share=True)