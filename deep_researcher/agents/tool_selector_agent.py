"""
Agent used to determine which specialized agents should be used to address knowledge gaps.

The Agent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

KNOWLEDGE GAP TO ADDRESS: <knowledge gap that needs to be addressed>
===========================================================

The Agent then:
1. Analyzes the knowledge gap to determine which agents are best suited to address it
2. Returns an AgentSelectionPlan object containing a list of AgentTask objects

The available agents are:
- WebSearchAgent: General web search for broad topics
- SiteCrawlerAgent: Crawl the pages of a specific website to retrieve information about it
"""

from pydantic import BaseModel, Field
from typing import List
from agents import Agent
from ..llm_client import fast_model
from datetime import datetime


class AgentTask(BaseModel):
    """A task for a specific agent to address knowledge gaps"""
    gap: str = Field(description="The knowledge gap being addressed")
    agent: str = Field(description="The name of the agent to use")
    query: str = Field(description="The specific query for the agent")
    entity_website: str = Field(description="The website of the entity being researched, if known")


class AgentSelectionPlan(BaseModel):
    """Plan for which agents to use for knowledge gaps"""
    tasks: List[AgentTask] = Field(description="List of agent tasks to address knowledge gaps")


INSTRUCTIONS = f"""
You are an Tool Selector responsible for determining which specialized agents should address a knowledge gap in a research project.
Today's date is {datetime.now().strftime("%Y-%m-%d")}.

You will be given:
1. The original user query
2. A knowledge gap identified in the research
3. A full history of the tasks, actions, findings and thoughts you've made up until this point in the research process

Your task is to decide:
1. Which specialized agents are best suited to address the gap
2. What specific queries should be given to the agents (keep this short - 3-6 words)

Available specialized agents:
- WebSearchAgent: General web search for broad topics (can be called multiple times with different queries)
- SiteCrawlerAgent: Crawl the pages of a specific website to retrieve information about it - use this if you want to find out something about a particular company, entity or product

Guidelines:
- Aim to call at most 3 agents at a time in your final output
- You can list the WebSearchAgent multiple times with different queries if needed to cover the full scope of the knowledge gap
- Be specific and concise (3-6 words) with the agent queries - they should target exactly what information is needed
- If you know the website or domain name of an entity being researched, always include it in the query
- If a gap doesn't clearly match any agent's capability, default to the WebSearchAgent
- Use the history of actions / tool calls as a guide - try not to repeat yourself if an approach didn't work previously
"""


tool_selector_agent = Agent(
    name="ToolSelectorAgent",
    instructions=INSTRUCTIONS,
    model=fast_model,
    output_type=AgentSelectionPlan,
)
