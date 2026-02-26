import os
from dotenv import load_dotenv
from langchain_core import embeddings
load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


def main():
    print("Hello from rag-gist!")
    print (os.environ['PINECONE_API_KEY'])
    loader = TextLoader("/Users/vibhakar/courses/langchain/langchain-course-vibhakar/project/rag-gist/Why-clean2.txt")
    
    
    document = loader.load()
  
    text_spliter = CharacterTextSplitter(separator='\n',chunk_size=1000, chunk_overlap=0)
    texts=text_spliter.split_documents(document)
    print(f"created  {len(texts)} chunks")
    embeddings = OllamaEmbeddings(model="embeddinggemma")
    print("ingesting..")
    PineconeVectorStore.from_documents(texts, embeddings,index_name=os.environ["INDEX_NAME"])
    print("finish..")
if __name__ == "__main__":
    main()
