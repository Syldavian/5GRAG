import gradio as gr

from controller import Controller
from langchain.chains import ConversationChain
from typing import Optional, Tuple
from threading import Lock

langchain_controller = Controller()
langchain_controller.createVectorStore()


def respond(prompt,history):
    resp = langchain_controller.runController(prompt,history)
    history.append((prompt,resp))
    return history


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

with gr.Blocks() as demo:
    toggleDB = gr.Button("toggle flag")
    toggleDB.click(fn=langchain_controller.toggleDatabase)
    chatbot=gr.Chatbot(height=500)
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7)
    textbox.submit(respond,inputs=[textbox,chatbot],outputs=chatbot)

demo.launch(share=True)