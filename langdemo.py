from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser #converts output into string
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS #simple local vector store
from langchain_text_splitters import RecursiveCharacterTextSplitter




output_parser = StrOutputParser()




# vector store part
# this lets you load the data you want to index into the vector store

embeddings = OpenAIEmbeddings() #This is for creating an embedding model to be used in a vector store
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide") # loading the document into a loader
docs = loader.load() 

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


#prompt template guides response
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

llm = ChatOpenAI(api_key = API_KEY, model='gpt-3.5-turbo-0125')

chain = prompt | llm | output_parser

resp = chain.invoke({"input":"how do I use langchain?"})
print(f"response is {resp}")



