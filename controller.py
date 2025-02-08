from settings import config
from DBClient import DBClient
from AutoFetcher import AutoFetcher
from utils import unzipFile
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser #converts output into string
from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever

import os

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
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large',api_key=API_KEY) #Since we're using openAI's llm, we have to use its embedding model
        
        self.db = DBClient(embedding_model=embeddings)
        endpoints = ["https://www.3gpp.org/ftp/Specs/latest/Rel-16/38_series","https://www.3gpp.org/ftp/Specs/latest/Rel-17/38_series","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series"]
        self.params = params = {"sortby":"date"}
        self.af = AutoFetcher(endpoints,unzipFile)

        self.isDatabaseTriggered = True

    def resyncDB(self):
        """Scan dir for new docs and add them"""
        #Fetch new docs
        print(f"resyncing on controller end")
        file_list = self.af.run(self.params)
        file_list = [file[:-4] + ".docx" for file in file_list]
        #split & break down new docs

        #update chroma
        self.db.updateDB(file_list)

        #reconstruct retriever

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
                            retriever=self.db.getRetriever(), llm=self.llm
                        ) 
        else:
            # If we have selected one or more docs, then apply filtering
            name_list = [os.path.join(DOC_DIR,doc) for doc in selected_docs]
            print("name_list: ", name_list)
            name_filter = {"source": {"$in": name_list}}
            return MultiQueryRetriever.from_llm(
                            retriever=self.db.getRetriever(search_kwargs={'filter': name_filter}), llm=self.llm
                        )            


    def runController(self, prompt, history, selected_docs):

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


