"""
Agent used to crawl a website and return the results.

The SearchAgent takes as input a string in the format of AgentTask.model_dump_json(), or can take a simple starting url string as input

The Agent then:
1. Uses the crawl_website tool to crawl the website
2. Writes a 3+ paragraph summary of the crawled contents
3. Includes citations/URLs in brackets next to information sources
4. Returns the formatted summary as a string
"""

from agents import Agent
from ...tools import crawl_website
from . import ToolAgentOutput
from ...llm_client import fast_model


INSTRUCTIONS = """
You are a web craling agent that crawls the contents of a website answers a query based on the crawled contents. Follow these steps exactly:

* From the provided information, use the 'entity_website' as the starting_url for the web crawler
* Crawl the website using the crawl_website tool
* After using the crawl_website tool, write a 3+ paragraph summary that captures the main points from the crawled contents
* In your summary, try to comprehensively answer/address the 'gaps' and 'query' provided (if available)
* If the crawled contents are not relevant to the 'gaps' or 'query', simply write "No relevant results found"
* Use headings and bullets to organize the summary if needed
* Include citations/URLs in brackets next to all associated information in your summary
* Only run the crawler once
"""

crawl_agent = Agent(
    name="SiteCrawlerAgent",
    instructions="You are a site crawler agent that crawls a website and returns the results. ",
    tools=[crawl_website],
    model=fast_model,
    output_type=ToolAgentOutput,
)
