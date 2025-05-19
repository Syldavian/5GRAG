from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils import RefObj
from ReferenceExtractor import ReferenceExtractor
import os 
from settings import config
from langchain_core.prompts import ChatPromptTemplate 

RExt = ReferenceExtractor()
DOC_DIR = config["DOC_DIR"]
NUM_EXTRA_DOCS = config["NUM_EXTRA_DOCS"]

class MultiStageRetriever:
    def __init__(self,llm,prompt_template):
        self.llm = llm
        self.document_prompt = ChatPromptTemplate.from_template("""
        Page content: {page_content} \n
        From clause: {section}
        """)
        self.doc_chain = create_stuff_documents_chain(self.llm, prompt_template, document_prompt=self.document_prompt)
        self.selected_docs = None

    def reconstructDocChain(self,new_prompt_template):
        self.doc_chain = create_stuff_documents_chain(self.llm,new_prompt_template,document_prompt=self.document_prompt)

    def constructRetriever(self,db,selected_docs=None):
        """@selected_docs: list of documents to filter.
        @db: DBClient instance.
        @llm: the llm used for rewriting queries
        returns: Nothing"""
        if selected_docs is None or len(selected_docs) == 0:
            self.base_retriever = MultiQueryRetriever.from_llm(
                            retriever=db.getRetriever(), llm=self.llm
                        ) 
            self.selected_docs = None
        else:
            # If we have selected one or more docs, then apply filtering
            name_list = [os.path.join(DOC_DIR,doc) for doc in selected_docs]
            print("name_list: ", name_list)
            name_filter = {"source": {"$in": name_list}}
            self.selected_docs = name_list
            self.base_retriever = MultiQueryRetriever.from_llm(
                            retriever=db.getRetriever(search_kwargs={'filter': name_filter}), llm=self.llm
                        )

    def getAdditionalContext(self,org_docs,db):
        """@org_docs: list of initially retrieved document chunks from vector db.
        In this method, we parse the org_docs for external references and perform additional retrievals.
        returns: list of document chunks"""
        ext_src: list[RefObj] = RExt.runREWithDocList(docs=org_docs)
        section_names = []
        section_names = RExt.extractClauseNumbersOfSrc(ext_src)
        print(f"\n extr_src is {ext_src}\n")
        print(f"section_names are {section_names}")
        if section_names == []:
            return []

        if self.selected_docs:
            metadata_filter = {'$and':[
                {'source':{'$in':self.selected_docs}},
                {'section':{"$in":section_names}}
            ]}
        else:    
            metadata_filter = {"section":{"$in":section_names}}
        
        #metadataOnlyRetriever = db.getRetriever(search_kwargs={'filter':metadata_filter,'k':1000})
        additional_docs = []
        additional_docs.extend(db.vector_db.similarity_search("",NUM_EXTRA_DOCS,filter=metadata_filter))

        return additional_docs

    def invoke(self,query,history,db):
        if not self.base_retriever:
            raise Exception("Error: No base retriever initialized. Has constructRetriever been run?")
        org_docs = self.base_retriever.invoke(query)
        print(f"org docs are {org_docs}")
        additional_docs = self.getAdditionalContext(org_docs,db)
        print(f"\n\n additional docs are {additional_docs} \n\n")


        retrieved_docs = org_docs + additional_docs

        resp_answer = self.doc_chain.invoke({"context":retrieved_docs,"input":query,"history":history})
        resp = {"input":query,"history":history,"context":retrieved_docs,"answer":resp_answer}
        return resp,org_docs,additional_docs
    
    