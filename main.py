import os
from dotenv import load_dotenv
load_dotenv()
from operator import itemgetter

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

print ("initializing componenets ...")

embeddings = OllamaEmbeddings(model="llama3.1")
llm = ChatOllama(model="llama3.1")
llm_option0=ChatOllama(model="gemma3")
vectorestore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=embeddings
)
retriever = vectorestore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context
    {context}
    Question: {question}
    Provide a detailed answer: 
    """
)

def create_retriveal_chain_with_lcel():
    """create a retrieval chain
    """
    retrieval_chain=(
        RunnablePassthrough.assign(
            context=itemgetter("question")|retriever| format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain
def format_docs(docs):
    """Formatted retrived docs nicely  into one document
    """
    return("\n\n".join(doc.page_content for doc in docs))

def retieve_rag_without_lcel(query:str):
    """ Simple retrieval from RAG . A naive approach.
    """ 
    docs = retriever.invoke(query)
    context = format_docs(docs)
    message = prompt_template.format_messages(context=context, question=query)
    response=llm.invoke(message)
    return response.content

def main():
    print("Hello from rag-retrieve-naive!")

    query = "What is Pinecone in machine learing"
    #=====================================
    #Option 0 : Raw invocation
    #=====================================
    #print("Raw invocation, no RAG")
    #result_raw=llm_option0.invoke([HumanMessage(content=query)])
    #print("Answer0:")
    #print(result_raw)

    #=====================================
    #Option 1 : Naive implementation without LCEL
    #=====================================
    #print("Answer from RAG")
    #result_rag_without_lcel=retieve_rag_without_lcel(query=query)
    #print("Answer1:")
    #print(result_rag_without_lcel)

    #=====================================
    #Option 2 : Naive implementation with LCEL
    #=====================================
    print("Answer from RAG")
    chain_with_lcel=create_retriveal_chain_with_lcel()
    result= chain_with_lcel.invoke({"question": query})
    print("Answer2:")
    print(result)

if __name__ == "__main__":
    main()

