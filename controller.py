from settings import config
from DBClient import DBClient
from AutoFetcher import AutoFetcher
from utils import unzipFile
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser #converts output into string
from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from MultiStageRetriever import MultiStageRetriever
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
        endpoints = ["https://www.3gpp.org/ftp/Specs/latest/Rel-16/38_series","https://www.3gpp.org/ftp/Specs/latest/Rel-17/38_series"]
        #endpoints += ["https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series"]
        self.params = params = {"sortby":"date"}
        self.af = AutoFetcher(endpoints,unzipFile)

        self.retriever = MultiStageRetriever(llm=self.llm,prompt_template = self.prompt)

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
        print(f"done resyncing")
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
        self.retriever.reconstructDocChain(self.prompt)
        return self.isDatabaseTriggered

    def convert_history(self, history):
        """This turns the 'history' of the frontend into a particular format, separating the human and ai messages."""
        message_objects = []
        for turn in history:
            message_objects.append(HumanMessage(content=turn[0]))
            message_objects.append(AIMessage(content=turn[1]))
        return message_objects
    

    def getResponseWithRetrieval(self,prompt,history):
        
        resp = self.retriever.invoke(query=prompt,history=history,db=self.db)
        """all_docs = resp['context'][:]
        ext_src: list[RefObj] = RExt.runREWithDocList(docs=all_docs)
        #print(f"ext_src is {ext_src[0].reference}")
        for refObj in ext_src:
            res = self.retriever.invoke(f"You are an expert retriever with access to a vector database. Parse through the database, and only return data from this section: {refObj.reference}")
            print(f"for ref {refObj.reference}, res is {res}")"""

        return resp

    def runController(self, prompt, history, selected_docs):

        print('Selected Docs: ', selected_docs)
        self.retriever.constructRetriever(db=self.db,selected_docs=selected_docs)

        if prompt:
            print(f"Ctrl + C to exit...")
            #doc_chain is a chain that lets you pass a document to the llm and it uses that to answer
            # retrieval chain passed the load of deciding what document to use to answer to the retriever.
            history = self.convert_history(history)
            if self.isDatabaseTriggered:
                resp = self.getResponseWithRetrieval(prompt,history)
                print(f"resp is {resp}")
                response = resp['answer']
            else:
                chain = self.prompt | self.llm
                resp = chain.invoke({"input":prompt,"history": history})
                response = resp.content
            #print(f"resp is {resp}")
            return response
    
if __name__ == "__main__":
    c = Controller()
    c.runController()


