import asyncio
import os
import ssl
from typing import Any, Dict, List
from unittest import result

import certifi
from dotenv import load_dotenv
from langchain_core import embeddings
from langchain_core.documents import Document
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain_tavily import TavilyMap, TavilyCrawl, TavilyExtract, tavily_crawl, tavily_extract, tavily_map
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import Colors, log_error, log_header, log_info, log_success, log_warning
import logger

load_dotenv()

# create a ssl certificate
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

embeddings = OllamaEmbeddings(model="llama3.1")

vectorstore=PineconeVectorStore(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings)
tavily_extract= TavilyExtract()
tavily_map=TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()




async def main():

    print (""" Main async function to orchestrate the whole process""")
    log_header("Doc ingestion has started")

    log_info(
        "Tavily crawl to read https://python.langchain.com",
        Colors.PURPLE
    )

    # crawl the site
    res= tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth" : 5,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })
    all_docs = [Document(page_content=result['raw_content'], metadata={"source":result['url']}) for result in res['results']]
    log_success(f"TavilyCrawl: succesfully crawled {len(all_docs)} URLs from langchain site")





if __name__=="__main__":
    asyncio.run(main())
