from zipfile import ZipFile
import os
import subprocess

def unzipFile(filepath,dest_dir):
    """filepath must be an absolute path. 
    Downloads all files from the zip, and clears out the zip file"""
    with ZipFile(filepath,'r') as zf:
        zf.extractall(path=dest_dir)
    
    os.remove(filepath)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def convertAllDocToDocx(input_dir:str,output_dir:str=None):
    """Converts all the .doc files in `input_dir` to .docx, and then removes the old docs.
    the .docx files will be stored in `output_dir` which defaults to the same as `input_dir`"""
    if not output_dir:
        output_dir = input_dir
    for filename in os.listdir(input_dir):
        if filename.endswith(".doc"):
            filepath = os.path.join(input_dir,filename)
            subprocess.run(['soffice','--headless','--convert-to','docx','--outdir',output_dir,filepath])
            os.remove(filepath)

class RefObj:
    def __init__(self,reference:str,src:str):
        self.reference = reference
        self.src = src
    
    def __repr__(self):
        return f'Reference: {self.reference}, Source Document: {self.src}'

if __name__ == "__main__":
    convertAllDocToDocx("data")
