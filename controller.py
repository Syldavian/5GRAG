from settings import config
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser #converts output into string
from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

import pytesseract
import os
import pickle

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\tesseract.exe"

API_KEY = config["API_KEY"]
M_NAME = config["MODEL_NAME"]
DOC_DIR = config["DOC_DIR"]

class Controller:
    def __init__(self):
        self.output_parser = StrOutputParser()
        self.llm = ChatOpenAI(api_key = API_KEY, model=M_NAME)
        self.prompt = ChatPromptTemplate.from_template("""Answer the following question with reference to the provided context:
<context>
{context}
</context>
Question: {input}""")
        self.docs = None
        self.isCreated = False
        self.isDatabaseTriggered = True
        self.vector = None

    def updateDocs(self):
        #We need a separate loader for each document. 
        self.docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True) #creates a text splitter, which breaks apart the document into text
        for file in os.listdir(DOC_DIR):
            loader = PyPDFLoader(os.path.join(DOC_DIR,file))
            print(f"Document is {loader.file_path}")
            # raw_doc = loader.load_and_split() 

            # Adam Chen Pickle Mode
            with open(os.path.join(DOC_DIR,file), 'rb') as handle:
                raw_doc = pickle.load(handle)    
            # End Pickle mode

            print("metadata: ")
            print(raw_doc[0].metadata)

            #print(raw_doc[:5])
            doc = text_splitter.split_documents(raw_doc) #applies the text splitter to the documents
            self.docs.extend(doc)

    def toggleDatabase(self):
        self.isDatabaseTriggered = not self.isDatabaseTriggered
        print(f"flag is {self.isDatabaseTriggered}")
        if self.isDatabaseTriggered:
            self.prompt = ChatPromptTemplate.from_template("""Answer the following question with reference to the provided context:
<context>
{context}
</context>
Question: {input}""")
        else:
            self.prompt = ChatPromptTemplate.from_template("""Answer the following question as best you can Question: {input}""")
        return self.isDatabaseTriggered

    def createVectorStore(self):
        """ create the vector database which will store the vector embeddings of the\
              documents that will be retrieved."""
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=API_KEY) #Since we're using openAI's llm, we have to use its embedding model
        self.updateDocs()
        self.vector = FAISS.from_documents(self.docs, embeddings) 
        self.retriever = MultiQueryRetriever.from_llm(
                            retriever=self.vector.as_retriever(), llm=self.llm
                        ) 
        
        self.isCreated = True

    def convert_history(self, history):
        message_objects = []
        for turn in history:
            message_objects.append(HumanMessage(content=turn[0]))
            message_objects.append(AIMessage(content=turn[1]))
        print(f"message objects are {message_objects}")
        return message_objects
    

    def runController(self, prompt, history, selected_docs):
        print(f"history is {history}")
        if not self.isCreated:
            self.createVectorStore()

        # If we have selected one or more docs, then apply filtering
        if len(selected_docs) > 0:
            name_list = ['./files/' + doc for doc in selected_docs]
            name_filter = {"source": {"$in": name_list}}
            self.retriever = MultiQueryRetriever.from_llm(
                            retriever=self.vector.as_retriever(search_kwargs={'filter': name_filter}), llm=self.llm
                        ) 


        
        if prompt:
            print(f"Ctrl + C to exit...")
            #doc_chain is a chain that lets you pass a document to the llm and it uses that to answer
            # retrieval chain passed the load of deciding what document to use to answer to the retriever.
            history = self.convert_history(history)
            if self.isDatabaseTriggered:
                doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
                retrieval_chain = create_retrieval_chain(self.retriever, doc_chain)
    
                resp = retrieval_chain.invoke({"input":prompt,"history": history})
                response = resp['answer']
            else:
                chain = self.prompt | self.llm
                resp = chain.invoke({"input":prompt,"history": history})
                response = resp.content
            print(f"resp is {resp}")
            return response
    
if __name__ == "__main__":
    c = Controller()
    c.runController()


