"""
Agent used to produce an initial outline of the report, including a list of section titles and the key question to be 
addressed in each section.

The Agent takes as input a string in the following format:
===========================================================
QUERY: <original user query>
===========================================================

The Agent then outputs a ReportPlan object, which includes:
1. A summary of initial background context (if needed), based on web searches and/or crawling
2. An outline of the report that includes a list of section titles and the key question to be addressed in each section
"""

from pydantic import BaseModel, Field
from typing import List
from agents import Agent
from ..llm_client import main_model
from .tool_agents.crawl_agent import crawl_agent
from .tool_agents.search_agent import search_agent
from datetime import datetime

class ReportPlanSection(BaseModel):
    """A section of the report that needs to be written"""
    title: str = Field(description="The title of the section")
    key_question: str = Field(description="The key question to be addressed in the section")


class ReportPlan(BaseModel):
    """Output from the Report Planner Agent"""
    background_context: str = Field(description="A summary of supporting context that can be passed onto the research agents")
    report_outline: List[ReportPlanSection] = Field(description="List of sections that need to be written in the report")


INSTRUCTIONS = f"""
You are a research manager, managing a team of research agents. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
Given a research query, your job is to produce an initial outline of the report (section titles and key questions),
as well as some background context. Each section will be assigned to a different researcher in your team who will then
carry out research on the section.

You will be given:
- An initial research query

Your task is to:
1. Produce 1-2 paragraphs of initial background context (if needed) on the query by running web searches or crawling websites
2. Produce an outline of the report that includes a list of section titles and the key question to be addressed in each section. 

Guidelines:
- Each section should cover a single topic/question that is independent of other sections
- The key question for each section should include both the NAME and DOMAIN NAME / WEBSITE (if available and applicable) if it is related to a company, product or similar
- The background_context should not be more than 2 paragraphs
- The background_context should be very specific to the query and include any information that is relevant for researchers across all sections of the report
- The background_context should be draw only from web search or crawl results rather than prior knowledge (i.e. it should only be included if you have called tools)
- For example, if the query is about a company, the background context should include some basic information about what the company does
"""

    
planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    tools=[
        search_agent.as_tool(
            tool_name="web_search",
            tool_description="Use this tool to search the web for information relevant to the query - provide a query with 3-6 words as input"
        ),
        crawl_agent.as_tool(
            tool_name="crawl_website",
            tool_description="Use this tool to crawl a website for information relevant to the query - provide a starting URL as input"
        )
    ],
    model=main_model,
    output_type=ReportPlan,
)