from dotenv import load_dotenv 
load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch



from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse


tools = [TavilySearch()]
llm = ChatOllama(model="llama3.1")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tool_names","input","agent_scrtachpad"]
).partial(format_instructions=output_parser.get_format_instructions())
react_agent = create_react_agent(llm=llm, tools = tools, prompt=react_prompt_with_format_instructions)
#agentExecutor = AgentExecutor(agent=react_agent, tools = tools, verbose=True, handle_parsing_errors=True)
agentExecutor = AgentExecutor(agent=react_agent, tools = tools, verbose=True)
extractOutput = RunnableLambda(lambda x: x["output"])
parseOutput = RunnableLambda(lambda x : output_parser.parser(x))
chain = agentExecutor|extractOutput|parseOutput

#chain = agentExecutor


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
