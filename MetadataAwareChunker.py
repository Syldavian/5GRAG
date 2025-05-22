from docx import Document as DocParser
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from collections.abc import Callable
from utils import getFirstPageOfDocxInMarkdown,getMetadataFromLLM

BASE_SECTION_NAME = "N/A"

def parse_table(table):
    """@input: docx table. Consists of rows and cells and the cells may have other tables.
    This is a helper function to help parse tables from docx. 
    It is needed as tables may contain nested tables."""
    for row_idx,row in enumerate(table.rows):
        for col_idx,col in enumerate(row.cells):
            if col.tables:
                for nested_table in col.tables:
                    parse_table(nested_table)
            else:
                yield col.text

def process_section_name(section_name:str)->str:
    """@input: (str)  section name.
    This is a helper function to process the section name into a numerical format.
    We do this to keep it consistent with how external references will be searched for.
    """
    return section_name.split("\t")[0]

def extract_core_properties(file)->dict:
    """This is a helper to pull out the core properties of a docx file"""
    core_obj = file.core_properties
    return {'author':core_obj.author,'title':core_obj.title,'subject':core_obj.subject}

def addExtraDocumentWideMetadataForContext(text_chunk:str,filepath:str):
    """Use this for 3gpp specs"""
    extractVersionAndDocIDRegx = re.compile(r"(3GPP TS (\d+.\d+|\-\d)+ V\d+.\d+.\d)",re.IGNORECASE)
    extractVersion = re.compile(r"(V\d+.\d+.\d)")
    extractDocument = re.compile(r"(TS (\d+.\d+|\-\d)+)")

    searchRes = extractVersionAndDocIDRegx.search(text_chunk)
    if searchRes:
        res = searchRes.group()
        version = extractVersion.search(res).group()[1:]
        docID = extractDocument.search(res).group()[3:]
    else:
        return {}
    
    metadata = {'version':version,'docID':docID}
    return metadata

def addExtraDocumentWideMetadataForReason(text_chunk:str,filepath:str):
    """Use this for TDocs"""
    mdData = getFirstPageOfDocxInMarkdown(filepath)
    print(f"\n\n mddata is {mdData} \n\n")
    metadata =getMetadataFromLLM(mdData)
    print(metadata)
    return metadata

def getSectionedChunks(file_list,addExtraDocumentWideMetadata:Callable[[str,str],dict]=addExtraDocumentWideMetadataForContext):
    """@input: file_list. List of files in relative path that will be chunked.
    @addExtraDocumentWideMetadata: func that returns a dictionary with extra metadata that will be added to all chunks.
    Returns: master list chunks_with_metadata that has chunks of all the files stored as langchain Documents.
    These Documents have section metadata"""
    chunks_with_metadata= []
    for file in file_list:

        f = open(file,'rb')
        doc = DocParser(f)

        file_metadata = extract_core_properties(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)

        current_section_title = BASE_SECTION_NAME
        current_section_text = ""
        sections = []
        #TODO: Switch to collecting a "current section" list of text chunks
        # currently there's no overlap between paragraphs in the same section which is bad for retrieval
        # We want to collect (section_text,current_section) where section_text is a combination of paragraphs
        # Then, we use the text_splitter to split it into split chunks
        for part in doc.iter_inner_content():
            #print(f"part is {part}")
            if isinstance(part,docx.text.paragraph.Paragraph) and part.style and part.style.name.startswith('Heading'):
                # update current section
                sections.append((current_section_text,current_section_title))
                current_section_text = ""
                current_section_title = part.text
            elif isinstance(part,docx.table.Table):
                for table_datum in parse_table(part):
                    current_section_text += table_datum
            else:
                # append text to current section
                current_section_text += part.text
        
        #Add last section to sections
        sections.append((current_section_text,current_section_title))

        addMetadata = {}

        for text,section_name in sections:
            split_chunks = text_splitter.split_text(text)
            
            if section_name == BASE_SECTION_NAME and not addMetadata:
                addMetadata = addExtraDocumentWideMetadata(text,file)
                print(f"metadata is {addMetadata}")

            for chunk in split_chunks:
                chunks_with_metadata.append(Document(
                    page_content=chunk,
                    metadata={'source':file,'section':process_section_name(section_name),**addMetadata,**file_metadata}
                ))
    return chunks_with_metadata

def getFullSectionChunks(file_list,addExtraDocumentWideMetadata:Callable[[str,str],dict]=addExtraDocumentWideMetadataForContext):
    chunks_with_metadata = []
    for file in file_list:
        f = open(file,'rb')
        doc = DocParser(f)

        file_metadata = extract_core_properties(doc)

        current_section_title = BASE_SECTION_NAME
        current_section_text = ""
        sections = []

        for part in doc.iter_inner_content():
            if isinstance(part,docx.text.paragraph.Paragraph) and part.style and part.style.name.startswith('Heading'):
                # update current section
                sections.append((current_section_text,current_section_title))
                current_section_text = ""
                current_section_title = part.text
            elif isinstance(part,docx.table.Table):
                for table_datum in parse_table(part):
                    current_section_text += table_datum
            else:
                # append text to current section
                current_section_text += part.text
        
        #Add last section to sections
        sections.append((current_section_text,current_section_title))

        addMetadata = {}
        for text,section_name in sections:
            if section_name == BASE_SECTION_NAME and not addMetadata:
                addMetadata = addExtraDocumentWideMetadata(text,file)
                print(f"metadata is {addMetadata}")
            
            chunks_with_metadata.append(Document(
                page_content=text,
                metadata = {'source':file,'section':process_section_name(section_name),**addMetadata,**file_metadata}
            ))

        return chunks_with_metadata

def getFullFileChunks(file_list):
    chunks = []
    for file in file_list:

        f = open(file,'rb')
        doc = DocParser(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text
        split_chunks = text_splitter.split_text(text)
        for chunk in split_chunks:
            chunks.append(Document(
                page_content=chunk,
                metadata={'source':file}
            ))
    return chunks

if __name__ =="__main__":
    """file_list = ["./data/38214-hc0.docx","./data/v16diffver.docx"]
    #print(getSectionedChunks(file_list)[80:90])
    sectioned_things = {}
    for doc in getFullSectionChunks([file_list[0]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc.page_content]

    for doc in getFullSectionChunks([file_list[1]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc.page_content]
    
    print(sectioned_things['2'])"""
    file_list = ["./data/TR 38.845 v0.1.0.docx"]
    #print(Docx2txtLoader(file_list[0]).load())
    print(getSectionedChunks(file_list,addExtraDocumentWideMetadataForReason)[:50])
    
    