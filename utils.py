from zipfile import ZipFile
import os

def unzipFile(filepath,dest_dir):
    """filepath must be an absolute path. 
    Downloads all files from the zip, and clears out the zip file"""
    with ZipFile(filepath,'r') as zf:
        zf.extractall(path=dest_dir)
    
    os.remove(filepath)

class RefObj:
    def __init__(self,reference:str,src:str):
        self.reference = reference
        self.src = src
    
    def __repr__(self):
        return f'Reference: {self.reference}, Source Document: {self.src}'
