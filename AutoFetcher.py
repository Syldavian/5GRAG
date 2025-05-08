from settings import config
import requests
from bs4 import BeautifulSoup
import os
import random
from utils import unzipFile
# idea:
# These pages are maintained quite consistently. 
# two questions:
# 1) How do we get the latest spec? How do we pass it into the data folder?
# 2) When do we pull and when do we not?

class AutoFetcher:
    def __init__(self,fetch_endpoints:list[str],post_processing_func):
        """@fetch_endpoints: list of the endpoints that the AF will query with a get request when `AF.run()` is invoked.\n
        @post_processing_func: the function that is run on the fetched files."""
        self.links = {}
        self.fetch_endpoints = fetch_endpoints
        self.post_processing_func = post_processing_func

    def extractLinksFromEndpoint(self,endpoint,params):
        """For a specific endpoint we retrieve the page and collect all the links on that page. 
        Run this for pages which are collections of links.
        @params: search parameters for the request. Try to sort all the links by upload date. 
        The assumption is that the last link after sorting is the most recent one."""
        response = requests.get(endpoint,params=params)
        print(response)
        if response.status_code != 200:
            raise Exception("Error: Could not retrieve page content")
        soup = BeautifulSoup(response.content,'html.parser')
        self.links[endpoint] = []
        for link in soup.find_all('a'):
            self.links[endpoint].append(link.get('href'))
    
    def getMostRecentLink(self,endpoint):
        """Current assumption is that the last link is the most recent one"""
        if not self.links.get(endpoint,[]):
            raise Exception("Error: Run extractLinksFromEndpoint first")
        return self.links[endpoint][-1]
    
    def downloadFileFromLink(self,link)->str:
        """Makes the assumption that the last part of the address given is the name of the file to the downloaded
        Note: currently set up to work with duplicates. If the filename already exists, it will append a random hash to the end of the filename and then try to download.
        @link: http[s] endpoint where a file can be downloaded with a get request
        returns: filename (not abs path)"""
        filename = link.split("/")[-1]
        filepath = os.path.join(config['DOC_DIR'],filename)
        while os.path.exists(filepath):
            filepath = os.path.join(config['DOC_DIR'],filename+f"{random.randint(0,65535)}")

        response = requests.get(link)
        if response.status_code != 200:
            raise Exception("Error: Could not retrieve page content")

        with open(filepath,'wb') as file:
            file.write(response.content)
        
        self.post_processing_func(filepath,config['DOC_DIR'])
        return filename

    def run(self,params=None):
        file_list = []
        for endpoint in self.fetch_endpoints:
            self.extractLinksFromEndpoint(endpoint,params)
            link = self.getMostRecentLink(endpoint)
            filename=self.downloadFileFromLink(link)
            file_list.append(filename)
        return file_list

if __name__ == "__main__":
    params = {"sortby":"date"}
    endpoints = ["https://www.3gpp.org/ftp/Specs/latest/Rel-16/38_series","https://www.3gpp.org/ftp/Specs/latest/Rel-17/38_series","https://www.3gpp.org/ftp/Specs/latest/Rel-18/38_series"]
    af = AutoFetcher(endpoints,unzipFile)
    af.run(params)
