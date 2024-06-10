import gradio as gr

from controller import Controller
from langchain.chains import ConversationChain
from typing import Optional, Tuple
from threading import Lock

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


# More improved UI Deprecated chatInterface ise
"""demo = gr.ChatInterface(
    langchain_controller.runController,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7),
    title="5G Expert",
    description="Ask any question",
    theme="soft",
#    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
#    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)
demo.launch(share=True)"""

# idea: design function for the submit. This function will have as inputs, prompt and history
# get history from chatbot. As output, have "", history so both q and answer will go into chatbot



docs = ['38211-i20.pdf', 'ts_138331v160100p.pdf', 'ts_138211v160200p.pdf', 'ts_138331v170700p.pdf']
with gr.Blocks() as demo:
    toggleDB = gr.Button("RAG ENABLED CURRENTLY")
    toggleDB.click(fn=adjustToggle,outputs = toggleDB)
    chatbot=gr.Chatbot(height=500)
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7)
    selected_docs = gr.Dropdown(choices=docs, multiselect=True)
    textbox.submit(respond,inputs=[textbox,chatbot,selected_docs],outputs=chatbot)

demo.launch(share=True)