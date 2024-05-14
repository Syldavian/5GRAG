import gradio as gr

from controller import Controller
from langchain.chains import ConversationChain
from typing import Optional, Tuple
from threading import Lock

langchain_controller = Controller()
langchain_controller.createVectorStore()
# More improved UI
gr.ChatInterface(
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
).launch(share=True)