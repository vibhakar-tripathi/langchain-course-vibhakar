# from graphlib import TopologicalSorter
from ctypes import Union
from dotenv import load_dotenv
from pydantic_core.core_schema import union_schema

load_dotenv()

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import render_text_description
from langchain_core.prompts import PromptTemplate

from langchain.tools import tool, BaseTool

from callback import AgentCallbackhandler

from langchain_classic.schema import AgentAction, AgentFinish
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str

from langchain_core.callbacks.base import BaseCallbackHandler

from langchain_ollama import ChatOllama

# from callbacks import AgentCallbackHandler


@tool
def get_text_length(text: str) -> int:
    """Returns the length of characters in the given text"""
    print(f"get_text_length of the entered text as :{text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tool_list: list[tool], tool_name: str) -> tool:
    for tool in tool_list:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Could not find the tool name {tool_name=}")


# def format_log_to_str()


def main():
    print("Hello from react-langchain!")
    # print (get_text_length("dog"))
    tool_list = [get_text_length]

    myTemplate = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: [{scratchpad}]

    """
    print(tool_list)
    myPrompt = PromptTemplate.from_template(template=myTemplate).partial(
        tools=render_text_description(tool_list),
        tool_names=", ".join([t.name for t in tool_list]),
    )
    myLlm = ChatOllama(
        model="gemma3",
        temperature=0,
        stop=["\nObservation"],
        callbacks=[AgentCallbackhandler()],
    )
    intermediate_steps = []
    myAgent = (
        {
            "input": lambda x: x["input"],
            "scratchpad": lambda x: format_log_to_str(x["scratchpad"]),
        }
        | myPrompt
        | myLlm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""

    while not isinstance(agent_step, AgentFinish):

        agent_step: Union[AgentAction, AgentFinish] = myAgent.invoke(
            {
                "input": "What is the length of the string DOG?",
                "scratchpad": intermediate_steps,
            }
        )
        # print(res)
        # print(type(res))
        print(agent_step)
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tool_list, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

        print(agent_step)
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)


if __name__ == "__main__":
    main()
