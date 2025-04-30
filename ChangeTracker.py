from difflib import Differ
from MetadataAwareChunker import getFullSectionChunks
from langchain_core.documents import Document

SENTENCE_SEPARATOR = ". "

class ChangeTracker:
    @staticmethod
    def getChanges(chunk1:Document,chunk2:Document)->dict: 
        """@chunk1 and @chunk2 are two Documents representing the same chunk across different versions.
        Assumption is that chunk2 is the later version.
        Returns: change object dictionary. Has keys 'from_version', 'to_version', 'changes'.
        """
        if "version" not in chunk1.metadata or "version" not in chunk2.metadata:
            raise Exception("Cannot diff as no version metadata in one or both of the chunks")
        text1,text2 = chunk1.page_content, chunk2.page_content
        from_version,to_version = chunk1.metadata["version"], chunk2.metadata["version"]

        change_obj = {'from_version':from_version,'to_version':to_version,'changes':{'add':[],'remove':[]}}
        for delta in Differ().compare(text1.split(SENTENCE_SEPARATOR),text2.split(SENTENCE_SEPARATOR)):
            if len(delta)>0 and delta[0] == "+":
                change_obj['changes']['add'].append(delta[1:])
            elif len(delta) >0 and delta[0] == '-':
                change_obj['changes']['remove'].append(delta[1:])

        return change_obj

if __name__ == "__main__":
    file_list = ["./data/v16diffver.docx","./data/38141-2-gk0.docx"]

    sectioned_things = {}
    for doc in getFullSectionChunks([file_list[0]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc]

    for doc in getFullSectionChunks([file_list[1]]):
        sectioned_things[doc.metadata["section"]] = sectioned_things.get(doc.metadata["section"],[]) + [doc]
    
    print(ChangeTracker.getChanges(sectioned_things["Foreword"][0],sectioned_things["Foreword"][1]))
    