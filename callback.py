from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt.chat_agent_executor import Prompt
from langchain_classic.schema import LLMResult


class AgentCallbackhandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        print(f" Prompt to the LLM was : ***\n{prompts[0]}")
        print("******")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f" LLM response : ***\n{response.generations[0][0].text}")
        print("******")
