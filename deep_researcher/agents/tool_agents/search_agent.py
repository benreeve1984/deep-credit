"""
Agent used to perform web searches and summarize the results.

The SearchAgent takes as input a string in the format of AgentTask.model_dump_json(), or can take a simple query string as input

The Agent then:
1. Uses the web_search tool to retrieve search results
2. Analyzes the retrieved information
3. Writes a 3+ paragraph summary of the search results
4. Includes citations/URLs in brackets next to information sources
5. Returns the formatted summary as a string

The agent can use either OpenAI's built-in web search capability or a custom
web search implementation based on environment configuration.
"""

import os
from agents import Agent, WebSearchTool
from ...tools import web_search
from ...llm_client import fast_model
from dotenv import load_dotenv
from . import ToolAgentOutput

load_dotenv()
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serper").lower()

INSTRUCTIONS = """You are a research assistant. Given an AgentTask, follow these steps:

* Convert the 'query' into an optimized SERP search term for Google, limited to 3-5 words
* If an 'entity_website' is provided, make sure to include the domain name in your optimized Google search term
* Enter the optimized search term into the web_search tool
* After using the web_search tool, write a 3+ paragraph summary that captures the main points from the search results
* In your summary, try to comprehensively answer/address the 'gap' provided (which is the objective of the search)
* If the search results are not relevant to the search term or do not address the 'gap', simply write "No relevant results found"
* Use headings and bullets to organize the summary if needed
* Include citations/URLs in brackets next to all associated information in your summary
* Do not make additional searches
"""

if SEARCH_PROVIDER == "openai":
    web_search_tool = WebSearchTool()
else:
    web_search_tool = web_search

search_agent = Agent(
    name="WebSearchAgent",
    instructions=INSTRUCTIONS,
    tools=[web_search_tool],
    model=fast_model,
    output_type=ToolAgentOutput
)