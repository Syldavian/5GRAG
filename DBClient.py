from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from settings import config
import pickle
import os 

class DBClient:
    def addDocsFromFilePath(self,file_list):
        """@file_list: list(str) of file names. Not absolute paths"""
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True) #creates a text splitter, which breaks apart the document into text
        for file in file_list:
            loader = Docx2txtLoader(os.path.join(config['DOC_DIR'],file))

            if not config['IS_PICKLE']:
                raw_doc = loader.load_and_split() 
            # Adam Chen Pickle Mode
            else:
                with open(os.path.join(config['DOC_DIR'],file), 'rb') as handle:
                    raw_doc = pickle.load(handle)    
            # End Pickle mode
            print("metadata: ")
            print(raw_doc[0].metadata)

            doc = text_splitter.split_documents(raw_doc) #applies the text splitter to the documents
            docs.extend(doc)
        return docs

    def constructBaseDB(self,embedding_model):
        """We must populate the db with the initial files in the `DOC_DIR` directory"""
        docs = self.addDocsFromFilePath(os.listdir(config['DOC_DIR']))
        return Chroma.from_documents(docs,embedding_model)
    
    def __init__(self,embedding_model):
        #construct chroma base db            
        self.vector_db = self.constructBaseDB(embedding_model)

    def updateDB(self,new_file_list):
        """@new_file_list: list(str) list of file names (not abs paths)
        Turn the new files in DOC_DIR into a list of documents and add them
        to the vector store."""
        new_docs = self.addDocsFromFilePath(new_file_list)
        self.vector_db.add_documents(new_docs)

    def getRetriever(self,search_kwargs=None):
        if search_kwargs == None:
            return self.vector_db.as_retriever()
        return self.vector_db.as_retriever(search_kwargs=search_kwargs)
    
