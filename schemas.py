from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    "Source URL"
    url:str =Field (description="The URL of the source")

class AgentResponse(BaseModel):
    " Schema for response from the agent"
    answer:str = Field(description= "Text response of the query")
    sources:List[Source] = Field(default_factory=List, description="List of URLs as sources")
