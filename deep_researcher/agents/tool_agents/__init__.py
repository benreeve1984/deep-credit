from pydantic import BaseModel, Field

class ToolAgentOutput(BaseModel):
    """Standard output for all tool agents"""
    output: str
    sources: list[str] = Field(default_factory=list)

from .search_agent import search_agent
from .crawl_agent import crawl_agent

TOOL_AGENTS = {
    "WebSearchAgent": search_agent,
    "SiteCrawlerAgent": crawl_agent,
}