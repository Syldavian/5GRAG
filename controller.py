from settings import config
from DBClient import DBClient
from AutoFetcher import AutoFetcher
from utils import unzipFile,convertAllDocToDocx
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
        
        self.contextDB = DBClient(embedding_model=embeddings)
        self.reasonDB = DBClient(embedding_model=embeddings,collection_name="reason")

        # 38 series (NR and NG-RAN general)
        print("38 series")
        endpoints = [
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.300",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.401",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.331",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.321",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.322",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.323",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.340"
        ]

        # F1 interface (CU-DU split)
        print("f1 interface")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.470",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.471",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.472",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.473"
        ]

        # E1 interface (CU-CP / CU-UP split)
        print("e1 interface")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.460",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.463"
        ]

        # Xn interface (gNB to gNB)
        print("xn interface")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.420",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.423"
        ]

        # NG interface (gNB to 5GC)
        print("ng interface")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.410",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.413",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.425"
        ]

        # Dual connectivity and multi-RAT
        print("37 series")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.340",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.324",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.460",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.471",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.472",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.473"
        ]

        # LTE (E-UTRAN RAN architecture)
        print("36 series")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.300",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.401",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.410",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.413",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.420",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.423",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.331",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.321",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.322",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.323"
        ]

        # 5G System architecture and core interaction
        print("23 series")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.501",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.401",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.236",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.251",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.316"
        ]

        # OAM and SON (Operations & Automation)
        print("32 series")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.508",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.509",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.511",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.541",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.102"
        ]

        # UTRAN (3G RAN Architecture - background)
        print("25 series")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.401",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.410",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.413",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.420",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.423"
        ]

        # Release overview and terminology
        print("21 series")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/21_series/21.905"
        ]

        # RAN Study Items (TRs on architecture and functional splits)
        print("study TRs")
        endpoints += [
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.801",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.806",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.816",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.832",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.874",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.912",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.913"
        ]

        self.params = params = {"sortby":"date"}
        self.af = AutoFetcher(endpoints,unzipFile)

        otherEndpoints = ["https://www.3gpp.org/ftp/TSG_RAN/WG2_RL2/TSGR2_01/Docs/zips"]
        self.afReason = AutoFetcher(otherEndpoints,unzipFile)

        self.retriever = MultiStageRetriever(llm=self.llm,prompt_template = self.prompt)

        self.isDatabaseTriggered = True

    def updateContextDB(self):
        """Scan dir for new docs and add them"""
        #Fetch new docs
        print(f"resyncing on controller end")
        file_list = self.af.run(self.params)
        convertAllDocToDocx(DOC_DIR)
        file_list = [file[:-4] + ".docx" for file in file_list]
        #split & break down new docs

        #update chroma
        self.contextDB.updateDB(file_list)
        print(f"done resyncing")
        #reconstruct retriever'
    
    def updateReasonDB(self):
        """Fetches latest tdocs and reads into the reason collection"""
        print(f"Hit the update reason!")
        file_list = self.afReason.run()
        convertAllDocToDocx(DOC_DIR)
        file_list = [file[:-4] + ".docx" for file in file_list]
        
        self.reasonDB.updateDB(file_list)
        print("updated collection!")

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
        resp = self.retriever.invoke(query=prompt,history=history,db=self.contextDB)
        return resp

    def runController(self, prompt, history, selected_docs):

        print('Selected Docs: ', selected_docs)
        self.retriever.constructRetriever(db=self.contextDB,selected_docs=selected_docs)

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
            return response
    
if __name__ == "__main__":
    c = Controller()
    c.runController()


