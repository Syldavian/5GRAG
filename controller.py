from settings import config
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser #converts output into string
from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever

import os
import pickle

API_KEY = config["API_KEY"]
M_NAME = config["MODEL_NAME"]
DOC_DIR = config["DOC_DIR"]
IS_PICKLE = config["IS_PICKLE"]

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
        """This method attempts to set up the class attribute `self.docs` with a list of documents.
        It uses a loader to turn each pdf file in DOC_DIR into a 'Document object', which has a 'metadata' attribute and a `page_content` attribute.
        It then uses load_and_split to break apart each document into chunks, also stored in Document objects."""
        #We need a separate loader for each document. 
        self.docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True) #creates a text splitter, which breaks apart the document into text
        for file in os.listdir(DOC_DIR):
            loader = PyPDFLoader(os.path.join(DOC_DIR,file))

            if not IS_PICKLE:
                raw_doc = loader.load_and_split() 
            # Adam Chen Pickle Mode
            else:
                with open(os.path.join(DOC_DIR,file), 'rb') as handle:
                    raw_doc = pickle.load(handle)    
            # End Pickle mode

            print("metadata: ")
            print(raw_doc[0].metadata)

            doc = text_splitter.split_documents(raw_doc) #applies the text splitter to the documents
            self.docs.extend(doc)

    def toggleDatabase(self):
        """Switches from RAG mode to non-RAG mode"""
        self.isDatabaseTriggered = not self.isDatabaseTriggered
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

        #Adam Chen Hotfix Use Local DB
        if IS_PICKLE:
            self.vector = Chroma(persist_directory="./Chroma", embedding_function=embeddings)
        else:
            self.vector = Chroma.from_documents(self.docs, embeddings) 
        
        self.isCreated = True

    def convert_history(self, history):
        """This turns the 'history' of the frontend into a particular format, separating the human and ai messages."""
        message_objects = []
        for turn in history:
            message_objects.append(HumanMessage(content=turn[0]))
            message_objects.append(AIMessage(content=turn[1]))
        return message_objects
    

    def constructRetriever(self,selected_docs):
        if selected_docs is None or len(selected_docs) == 0:
            return MultiQueryRetriever.from_llm(
                            retriever=self.vector.as_retriever(), llm=self.llm
                        ) 
        else:
            # If we have selected one or more docs, then apply filtering
            name_list = [os.path.join(DOC_DIR,doc) for doc in selected_docs]
            print("name_list: ", name_list)
            name_filter = {"source": {"$in": name_list}}
            return MultiQueryRetriever.from_llm(
                            retriever=self.vector.as_retriever(search_kwargs={'filter': name_filter}), llm=self.llm
                        )            


    def runController(self, prompt, history, selected_docs):
        if not self.isCreated:
            self.createVectorStore()

        print('Selected Docs: ', selected_docs)
        self.retriever = self.constructRetriever(selected_docs)

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


