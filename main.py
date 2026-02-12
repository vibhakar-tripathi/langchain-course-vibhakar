from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
#from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import  TavilySearch

class Source(BaseModel):
    """Schema for url sources"""
    url:str = Field ("The URL of the source")
class AgentResponse(BaseModel):
    """Answer provided by the search including sources"""
    answer:str = Field(description="Agents answer to the query")
    sources:List[Source] = Field (default_factory=list, description="List of sources")
tavily = TavilyClient()

@tool
def search(query:str) -> str:
    """
    tool that searches over internet
    Args:
        query string
    result
        search result
    """
    print(f"searching for {query}")
    return (tavily.search(query=query))
    #return "Tokyo weather is sunny"
#llm = ChatOpenAI()
#llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm=ChatOllama(model="llama3.1")
#tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)

def main():
    print("Hello from react-search-agent!")
    #result = agent.invoke ({"messages": HumanMessage(content="How is the weather in Tokyo")})
    try:
        result = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs in bay area for AI Engineer position using langchain on LinkedIn and provide details")})
        print(result)
        print("Printing more !")
        #print(result['ToolMessage'])
    except Exception as e:
        print(f"Agent invoke failed: {e}")
        raise


if __name__ == "__main__":
    main()

