"""
Agent used to produce an initial outline of the report, including a list of section titles and the key question to be 
addressed in each section.

The Agent takes as input a string in the following format:
===========================================================
QUERY: <original user query>
===========================================================

The Agent then outputs a ReportPlan object, which includes:
1. A summary of initial background context on the target company
2. An outline of the report that includes a list of section titles and the key question to be addressed in each section
"""

from pydantic import BaseModel, Field
from typing import List
from agents import Agent
from ..llm_client import main_model
from .tool_agents.crawl_agent import crawl_agent
from .tool_agents.search_agent import search_agent
from datetime import datetime
from .prompt_constants import COMMON_GUIDELINES, RESEARCH_GUIDELINES, OUTPUT_FORMAT_GUIDELINES

class ReportPlanSection(BaseModel):
    """A section of the report that needs to be written"""
    title: str = Field(description="The title of the section")
    key_question: str = Field(description="The key question to be addressed in the section")


class ReportPlan(BaseModel):
    """Output from the Report Planner Agent"""
    background_context: str = Field(description="A summary of supporting context about the target company")
    report_outline: List[ReportPlanSection] = Field(description="List of sections that need to be written in the report")


INSTRUCTIONS = f"""\
You are a Senior Credit Analyst at a leading rating agency. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
Your job is to create a structured credit rating report outline that will lead to a single final credit rating for a company's senior unsecured debt.

Input:
- Credit rating research query with target company name

Tasks:
1. Immediately use web search/crawl tools to gather essential background information about the target company
2. Create a report outline based on the standard credit rating methodology provided below

Credit Rating Methodology Structure:
1. Business Risk Profile (30%)
   - Scale & Stability of Operations
   - Competitive Environment
   - Management & Strategy

2. Industry & Macro Risk (10%)
   - Industry Cyclicality & Trends
   - Macroeconomic Conditions

3. Financial Risk Profile (40%)
   - Profitability & Cash Flow
   - Leverage & Coverage
   - Historic & Projected Trends

4. Capital Structure & Liquidity (10%)
   - Capital Structure Position
   - Liquidity & Funding
   - Covenants & Protective Terms

5. Qualitative Modifiers (10%)
   - ESG & Governance Factors
   - Event Risks
   - Peer Group Positioning

Section Guidelines:
- Focus on researching the target company and its peers, not the rating methodology itself
- Each section should address a specific component of the credit analysis
- Include specific entity names and financial metrics needed in your key questions
- Background context should focus on company basics: business model, size, history, etc.

{OUTPUT_FORMAT_GUIDELINES}
{RESEARCH_GUIDELINES}
{COMMON_GUIDELINES}
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