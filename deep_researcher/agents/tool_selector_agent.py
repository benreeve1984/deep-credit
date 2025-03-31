"""
Agent used to determine which specialized agents should be used to address knowledge gaps.

The Agent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

KNOWLEDGE GAP TO ADDRESS: <knowledge gap that needs to be addressed>

GAP PRIORITY: <priority level of this gap (1-5)>

PREVIOUSLY ATTEMPTED: <whether this gap has been attempted before>
===========================================================

The Agent then:
1. Analyzes the knowledge gap to determine which agents are best suited to address it
2. Adapts its approach based on priority and whether the gap was previously attempted
3. Returns an AgentSelectionPlan object containing a list of AgentTask objects

The available agents are:
- WebSearchAgent: General web search for broad topics
- SiteCrawlerAgent: Crawl the pages of a specific website to retrieve information about it
"""

from pydantic import BaseModel, Field
from typing import List
from agents import Agent
from ..llm_client import fast_model
from datetime import datetime
from ..memory import MemoryManager


class AgentTask(BaseModel):
    """A task for a specific agent to address knowledge gaps"""
    gap: str = Field(description="The knowledge gap being addressed")
    agent: str = Field(description="The name of the agent to use")
    query: str = Field(description="The specific query for the agent")
    entity_website: str = Field(description="The website of the entity being researched, if known")


class AgentSelectionPlan(BaseModel):
    """Plan for which agents to use for knowledge gaps"""
    tasks: List[AgentTask] = Field(description="List of agent tasks to address knowledge gaps")


# Initialize memory manager
memory_manager = MemoryManager()


def get_relevant_memory(query: str, k: int = 3) -> str:
    """Retrieve relevant information from memory to help with agent selection."""
    results = memory_manager.query_chunks(query, k)
    if not results:
        return ""
    
    # Format relevant information for the prompt
    memory_text = "\nRelevant information from previous research:\n"
    for text, metadata in results:
        source = metadata.get('source', 'unknown')
        memory_text += f"\nFrom {source}:\n{text}\n"
    return memory_text


INSTRUCTIONS = f"""
You are an Adaptive Research Tool Selector responsible for determining which specialized agents should address knowledge gaps in a credit rating research project.
Today's date is {datetime.now().strftime("%Y-%m-%d")}.

You will be given:
1. The original user query
2. A knowledge gap identified in the research
3. The priority level of this gap (1=highest, 5=lowest)
4. Whether this gap has been previously attempted
5. A full history of the tasks, actions, findings and thoughts you've made up until this point in the research process
6. Relevant information from previous research (if available)

Your task is to decide:
1. Which specialized agents are best suited to address the gap
2. What specific queries should be given to the agents
3. How to adapt your approach based on priority and previous attempts

Available specialized agents:
- WebSearchAgent: General web search for broad topics (can be called multiple times with different queries)
- SiteCrawlerAgent: Crawl the pages of a specific website to retrieve information about it - use this if you want to find out something about a particular company, entity or product

Adaptive Guidelines:
- For first-time attempts on high-priority gaps (1-2), use precise, targeted queries
- For previously attempted gaps, try different approaches:
  * Use broader or more general search terms
  * Try alternative data sources or metrics
  * Look for proxy information or industry benchmarks
  * Search for qualitative assessments when quantitative data is unavailable
- For lower priority gaps (3-5), use broader queries to capture general information
- When a gap has been attempted multiple times, focus on getting partial information rather than perfect information

Query Construction Guidelines:
- Be appropriately detailed in your queries:
  * For simple information needs, use concise queries (3-6 words)
  * For complex financial or company-specific information, use more detailed queries
  * Include specific financial terms (debt-to-EBITDA, interest coverage, etc.)
- For previously attempted gaps, try these variations:
  * Add "estimate" or "approximate" to find analyst estimates
  * Try "[company] financial health" instead of specific metrics
  * Search for industry comparisons instead of company-specific data
  * Look for recent news or analyst opinions when hard data is unavailable
- When data is likely proprietary or limited, seek expert opinions or analyst reports instead

Tool Selection Logic:
- Use WebSearchAgent for broad information gathering and financial data
- Use SiteCrawlerAgent when you know the specific website that might contain the information
- For previously attempted gaps, try different combinations of tools with varied query approaches
- Aim to call at most 3 agents at a time in your final output
"""


tool_selector_agent = Agent(
    name="ToolSelectorAgent",
    instructions=INSTRUCTIONS,
    model=fast_model,
    output_type=AgentSelectionPlan,
)
