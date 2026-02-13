from dotenv import load_dotenv
load_dotenv()
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
#from langchain.agents.react.agent import 

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
tools = [TavilySearch()]

llm = ChatOllama(model="llama3.1")
react_prompt = hub.pull("hwchase17/react")

react_agent = create_react_agent(llm=llm, tools = tools, prompt=react_prompt, stream=True )
agentExecutor = AgentExecutor(agent=react_agent, tools = tools, verbose=True)
chain = agentExecutor


def main():
    print("Hello from react-search-agent!")
    result = chain.invoke(
        input = { 
            "input" : "Search for 3 jobs in bay area for AI Engineer position using langchain on LinkedIn and provide details"
        }
    )
    print (result)


if __name__ == "__main__":
    main()
