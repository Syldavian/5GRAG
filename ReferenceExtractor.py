import re
from utils import RefObj
from langchain_core.documents import Document

SRC_DOC = "Current_Doc"

class ReferenceExtractor:
    def __init__(self):
        self.regxs=[]
        self.extractDocRegx = re.compile(r"\[\d+, [A-Za-z0-9_\. ]*\]",re.IGNORECASE)
        patterns = [r"(clause\s+(\d+(.\d+)*).?(of \[\d+, [A-Za-z0-9_\. ]*\])?)",r"(Table\s+(\d+([.\d+|\-\d])*).?)"]
        for pattern in patterns:
            #re.compile turns a string into a regex. 
            self.regxs.append(re.compile(pattern,re.IGNORECASE))

    def findAllMatches(self,doc:str)->list[str]:
        """@doc: the string that we compare against all regexes
        returns: a list of the substrings that match any of the regexes"""
        matchedStrings = []
        for regx in self.regxs:
            temp = regx.findall(doc)
            if temp:
                for tup in temp:
                    matchedStrings.append(tup[0]) 
        return matchedStrings

    def extractDocumentFromStrings(self,matchedStrings:list[str])->list[RefObj]:
        """@matchedStrings: list of strings that were matched against the regexes.
        These were extracted from the original texts.
        This function will attempt to pull out the source document from this string if available.
        If not available, it will assign the current doc as source
        Returns: list[RefObj]"""
        references:list[RefObj] = []

        for matchedStr in matchedStrings:
            match = self.extractDocRegx.search(matchedStr)
            if match:
                src = match.group()
            else:
                src = SRC_DOC
            refWithoutSrc = matchedStr.replace(f" of {src}","")
            references.append(RefObj(reference=refWithoutSrc,src=src))
        return references

    def runREWithDocList(self,docs:list[Document])->list[RefObj]:
        """@docs: list of Documents, where Document is from langchain.
        Document: {metadata:{},page_content:,start_index:}
        Returns: a list of RefObj where each RefObj represents a reference found in the list of docs.
        RefObj: {reference:,src:}
        Note that as of right now, there may be duplicate RefObjs"""
        allRefs = []
        for doc in docs:
            matchedStrings = self.findAllMatches(doc.page_content)
            references = self.extractDocumentFromStrings(matchedStrings)
            allRefs.extend(references)
        return allRefs
    
    def extractClauseNumbersOfSrc(self,refs:list[RefObj])->list[str]:
        results = set()
        patterns = [r"(\d+(.\d+)*)",r"(\d+([.\d+|\-\d])*)"]
        extractRegxs = [ re.compile(pattern) for pattern in patterns]
        for ref in refs:
            if ref.src == SRC_DOC:
                for regx in extractRegxs:
                    temp = regx.findall(ref.reference)
                    if temp:
                        for tup in temp:
                            if len(tup)>0 and tup[-1] != ".":
                                results.add(tup[0]) 
        return list(results)


if __name__ == "__main__":
    examples = ["The determination of the used resource allocation table is defined in clause 6.1.2.1.1 of [4, TS 38.211] though you can also check Clause 6.2 or clause 6.3.",
    "Aperiodic CSI-RS is configured and triggered/activated as described in Clause 8.5.1.2", 
    "The UE shall derive CQI as specified in clause 5.2.2.1 of [TS, ]", "the UE procedure for receiving the PDSCH upon detection of a PDCCH follows clause 5.1 and the QCL assumption for the PDSCH as defined in clause 5.1.5"]
    re = ReferenceExtractor()
    matchedStrings = re.findAllMatches(examples[2])
    print(re.extractDocumentFromStrings(matchedStrings))