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
from ...memory import MemoryManager


# Initialize memory manager
memory_manager = MemoryManager()

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

def store_crawl_results(summary: str, url: str, crawled_urls: list[str]):
    """Store crawled results in the memory manager."""
    metadata = {
        'text': summary,
        'type': 'web_crawl',
        'source_url': url,
        'crawled_urls': crawled_urls,
        'source': 'crawl_agent'
    }
    return memory_manager.store_chunk(summary, metadata)

def get_relevant_crawl_results(query: str, k: int = 3) -> list[tuple[str, dict]]:
    """Retrieve relevant crawled content from memory."""
    return memory_manager.query_chunks(query, k)

crawl_agent = Agent(
    name="SiteCrawlerAgent",
    instructions=INSTRUCTIONS,
    tools=[crawl_website],
    model=fast_model,
    output_type=ToolAgentOutput,
)
