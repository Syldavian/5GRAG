from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from MetadataAwareChunker import getSectionedChunks,addExtraDocumentWideMetadataForReason
import chromadb
from settings import config
import os 
class DBClient:
    def addDocsFromFilePath(self,file_list):
        """@file_list: list(str) of file names. Not absolute/relative paths"""
        docs = []
        file_list = [os.path.join(config['DOC_DIR'],file) for file in file_list]
        if self.metadata_func:
            docs = getSectionedChunks(file_list,addExtraDocumentWideMetadata=self.metadata_func)
        else:
            docs = getSectionedChunks(file_list)
        
        return docs

    def constructBaseDB(self,embedding_model,collection_name):
        """Connect to the underlying chromadb"""
        #docs = self.addDocsFromFilePath(os.listdir(config['DOC_DIR']))
        pers_client = chromadb.PersistentClient(path=config["CHROMA_DIR"])
        vector_db = Chroma(client=pers_client,collection_name=collection_name,embedding_function=embedding_model)

        return vector_db
    
    def __init__(self,embedding_model,collection_name=config["SPEC_COLL_NAME"]):
        #construct chroma base db            
        self.vector_db = self.constructBaseDB(embedding_model,collection_name=collection_name)
        if collection_name == config["TDOC_COLL_NAME"]:
            self.metadata_func = addExtraDocumentWideMetadataForReason
        else:
            self.metadata_func = None

    def updateDB(self, new_file_list, batch_size=5000):
        """Update the vector DB in batches to avoid exceeding Chroma limits."""
        new_docs = self.addDocsFromFilePath(new_file_list)

        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i:i + batch_size]
            self.vector_db.add_documents(batch)

        print(f"Added {len(new_docs)} documents in batches.")

    def delFromDB(self):
        """This is a placeholder function. 
        Whilst doing this definitely prevents the retriever from fetching things with these docs as source,
        It still seems able to answer questions based on it."""
        self.vector_db.delete(where={'source':{'$eq':'data/38214-hc0.docx'}})
        self.vector_db.delete(where={'source':{'$eq':'data/38176-2-gc0.docx'}})
        self.vector_db.delete(where={'source':{'$eq':'data/38741-i31.docx'}})

    def getRetriever(self,search_kwargs=None):
        if search_kwargs == None:
            return self.vector_db.as_retriever()
        return self.vector_db.as_retriever(search_kwargs=search_kwargs)
    
if __name__ == "__main__":
    file_list = ["38214-hc0.docx"]
    print(DBClient.addDocsFromFilePath(file_list))