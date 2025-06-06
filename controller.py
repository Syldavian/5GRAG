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
SPEC_COLL_NAME = config["SPEC_COLL_NAME"]
TDOC_COLL_NAME = config["TDOC_COLL_NAME"]
TESTCASE_COLL_NAME = config["TESTCASE_COLL_NAME"]

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
        self.reasonDB = DBClient(embedding_model=embeddings,collection_name=TDOC_COLL_NAME)
        self.testcaseDB = DBClient(embedding_model=embeddings, collection_name=TESTCASE_COLL_NAME)

        endpoints = [
            # 21 series
            "https://www.3gpp.org/ftp/Specs/archive/21_series/21.905",

            # 23 series
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.003",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.007",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.032",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.041",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.203",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.216",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.236",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.247",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.251",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.287",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.304",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.316",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.401",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.501",
            "https://www.3gpp.org/ftp/Specs/archive/23_series/23.502",

            # 24 series
            "https://www.3gpp.org/ftp/Specs/archive/24_series/24.501",

            # 25 series
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.401",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.410",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.413",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.420",
            "https://www.3gpp.org/ftp/Specs/archive/25_series/25.423",

            # 26 series
            "https://www.3gpp.org/ftp/Specs/archive/26_series/26.114",
            "https://www.3gpp.org/ftp/Specs/archive/26_series/26.118",
            "https://www.3gpp.org/ftp/Specs/archive/26_series/26.247",

            # 28 series
            "https://www.3gpp.org/ftp/Specs/archive/28_series/28.405",

            # 29 series
            "https://www.3gpp.org/ftp/Specs/archive/29_series/29.244",
            "https://www.3gpp.org/ftp/Specs/archive/29_series/29.281",
            "https://www.3gpp.org/ftp/Specs/archive/29_series/29.510",
            "https://www.3gpp.org/ftp/Specs/archive/29_series/29.531",
            "https://www.3gpp.org/ftp/Specs/archive/29_series/29.571",

            # 32 series
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.102",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.422",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.508",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.509",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.511",
            "https://www.3gpp.org/ftp/Specs/archive/32_series/32.541",

            # 33 series
            "https://www.3gpp.org/ftp/Specs/archive/33_series/33.401",
            "https://www.3gpp.org/ftp/Specs/archive/33_series/33.501",

            # 36 series
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.104",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.211",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.300",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.304",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.306",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.314",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.321",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.322",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.323",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.331",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.401",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.410",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.413",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.420",
            "https://www.3gpp.org/ftp/Specs/archive/36_series/36.423",

            # 37 series
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.213",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.320",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.324",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.340",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.355",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.460",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.471",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.472",
            "https://www.3gpp.org/ftp/Specs/archive/37_series/37.473",

            # 38 series
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.101-1",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.104",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.133",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.211",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.213",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.214",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.215",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.300",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.304",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.305",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.314",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.321",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.322",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.323",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.331",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.340",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.401",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.410",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.413",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.414",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.420",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.423",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.425",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.455",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.460",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.463",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.470",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.471",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.472",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.473",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.801",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.806",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.816",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.832",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.874",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.912",
            "https://www.3gpp.org/ftp/Specs/archive/38_series/38.913"
        ]

        self.params = {"sortby":"date"}
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
    
    def updateContextDBlocal(self):
        """Scan dir for existing docs and add them to the database without fetching"""
        print(f"resyncing on controller end (using existing files)")
        
        # Get all existing docx files from the document directory
        file_list = []
        for filename in os.listdir(DOC_DIR):
            if filename.endswith('.docx'):
                file_list.append(filename)
        
        # Convert any remaining .doc files to .docx if needed
        convertAllDocToDocx(DOC_DIR)
        
        # Check again for any newly converted files
        for filename in os.listdir(DOC_DIR):
            if filename.endswith('.docx') and filename not in file_list:
                file_list.append(filename)
        
        additional_files = ["O-RAN.WG5.TS.IOT.0-R004-v12.00.docx", "ts_10392002v010101p.docx"]
        for file in additional_files:
            if file in file_list and os.path.exists(os.path.join(DOC_DIR, file)):
                file_list.remove(file)
        
        print(f"Found {len(file_list)} DOCX files to process: {file_list}")
        
        # Update chroma database with all found files
        if file_list:
            self.contextDB.updateDB(file_list)
            print(f"done resyncing - processed {len(file_list)} files")
        else:
            print("No DOCX files found to process")

    def updateReasonDB(self):
        """Fetches latest tdocs and reads into the reason collection"""
        print(f"Hit the update reason!")
        file_list = self.afReason.run()
        convertAllDocToDocx(DOC_DIR)
        file_list = [file[:-4] + ".docx" for file in file_list]
        
        self.reasonDB.updateDB(file_list)
        print("updated collection!")
    
    def updateTestcaseDBlocal(self):
        """Scan dir for existing docs and add them to the database without fetching"""
        print(f"resyncing on controller end (using existing files)")
        # Get all existing docx files from the document directory
        testcase_files = ["O-RAN.WG5.TS.IOT.0-R004-v12.00.docx", "ts_10392002v010101p.docx"]
        file_list = []
        for filename in testcase_files:
            if os.path.exists(os.path.join(DOC_DIR, filename)):
                file_list.append(filename)
        print(f"Found {len(file_list)} DOCX files to process: {file_list}")
        # Update chroma database with all found files
        if file_list:
            self.testcaseDB.updateDB(file_list)
            print(f"done resyncing - processed {len(file_list)} files")
        else:
            print("No DOCX files found to process")

    def switchDatabase(self, db_name):
        """Switch which database to use for retrieval
        
        Args:
            db_name (str): "context", "reason", or "testcase"
        """
        if db_name in ["context", "reason", "testcase"]:
            self.current_db = db_name
            print(f"Switched to {db_name} database")
        else:
            print(f"Invalid database name: {db_name}. Valid options: context, reason, testcase")
        return self.current_db

    def getCurrentDB(self):
        """Get the currently active database instance"""
        if self.current_db == "context":
            return self.contextDB
        elif self.current_db == "reason":
            return self.reasonDB
        elif self.current_db == "testcase":
            return self.testcaseDB
        else:
            return self.contextDB  # Default fallback

    def toggleDatabase(self):
        """Switches from RAG mode to non-RAG mode"""
        self.isDatabaseTriggered = not self.isDatabaseTriggered
        if self.isDatabaseTriggered:
            self.prompt = ChatPromptTemplate.from_template("""Answer the following question with reference to the provided context:
<context>
{context}
</context>
Question: {input}""")
            self.retriever.reconstructDocChain(self.prompt)
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
    
    def getResponseWithRetrieval(self,prompt,history):
        # Use the currently selected database
        current_db = self.getCurrentDB()
        resp,orig_docs,additional_docs = self.retriever.invoke(query=prompt,history=history,db=current_db)
        return resp,orig_docs,additional_docs

    def runController(self, prompt, history, selected_docs):
        print('Selected Docs: ', selected_docs)
        # Use the currently selected database
        current_db = self.getCurrentDB()
        self.retriever.constructRetriever(db=current_db,selected_docs=selected_docs)

        if prompt:
            print(f"Ctrl + C to exit...")
            #doc_chain is a chain that lets you pass a document to the llm and it uses that to answer
            # retrieval chain passed the load of deciding what document to use to answer to the retriever.
            history = self.convert_history(history)
            if self.isDatabaseTriggered:
                resp,orig_docs,additional_docs = self.getResponseWithRetrieval(prompt,history)
                #print(f"resp is {resp}")
                response = resp['answer']
            else:
                chain = self.prompt | self.llm
                resp = chain.invoke({"input":prompt,"history": history})
                response = resp.content
                orig_docs,additional_docs = [],[]
            return response,orig_docs,additional_docs
    
if __name__ == "__main__":
    c = Controller()
    #c.runController()
    c.updateContextDBlocal()
    c.updateTestcaseDBlocal()


