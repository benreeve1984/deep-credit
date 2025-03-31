"""
Agent used to produce the final draft of a report given initial drafts of each section.

The Agent takes as input the original user query and a stringified object of type ReportDraft.model_dump_json() (defined below).

====
QUERY:
{query}

REPORT DRAFT:
{report_draft}
====

The Agent then outputs the final markdown for the report.
"""

from pydantic import BaseModel, Field
from typing import List
from agents import Agent
from ..llm_client import reasoning_model
from datetime import datetime
from .prompt_constants import COMMON_GUIDELINES, OUTPUT_FORMAT_GUIDELINES, CITATION_FORMAT, QUALITY_GUIDELINES


class ReportDraftSection(BaseModel):
    """A section of the report that needs to be written"""
    section_title: str = Field(description="The title of the section")
    section_content: str = Field(description="The content of the section")


class ReportDraft(BaseModel):
    """Output from the Report Planner Agent"""
    sections: List[ReportDraftSection] = Field(description="List of sections that are in the report")


CREDIT_RATING_STYLE_GUIDE = """\
Credit Rating Report Style Requirements:

1. Document Structure:
   - Begin with a clear title: "Credit Rating Report: [Company Name]"
   - Executive Summary with final rating and outlook
   - Main body sections with detailed analysis
   - Conclusion restating the rating decision
   - References section

2. Narrative Style:
   - Use full, detailed paragraphs (not bullet points) for main analysis
   - Each paragraph should be 3-5 sentences minimum
   - Maintain academic, analytical tone throughout
   - Connect paragraphs with logical transitions
   - Preserve all numerical data and specific metrics
   - Ensure thorough explanation of rating rationale

3. Formatting:
   - Use H1 for main title, H2 for major sections, H3 for subsections
   - Use tables for financial data comparison where appropriate
   - Bold important conclusions and rating decisions
   - Maintain consistent terminology throughout

4. Length and Detail:
   - Final report should be minimum 8-10 pages of substantive content
   - Preserve all specific financial metrics and numerical data
   - Maintain detailed analysis in each section
   - Include multi-year trends and comparative analysis
"""


INSTRUCTIONS = f"""\
You are a Senior Credit Rating Report Editor. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
Your job is to polish and finalize credit rating reports while enhancing their professional quality and ensuring all essential analysis is preserved.

Input:
1. Original credit rating query
2. First draft of report sections

Tasks:
1. Combine sections into a single cohesive credit rating report
2. Format with proper headings and structure per credit rating style guide
3. Convert any bullet points into full, detailed paragraphs with narrative flow
4. Ensure all financial metrics and specific data points are preserved
5. Add an executive summary with the final credit rating and outlook
6. Organize content to follow standard credit rating report structure
7. Preserve and organize all sources/references
8. Ensure the final report is comprehensive with detailed analysis

Guidelines:
- Expand bullet points into full paragraphs with detailed analysis
- Preserve all numerical data and specific metrics
- Maintain narrative style with logical flow between sections
- Ensure paragraphs provide thorough explanation and context
- Follow credit rating report style guide for formatting and structure
- Keep all relevant financial analysis and comparative data
- Focus on professional, detailed narrative rather than brief summaries

{CREDIT_RATING_STYLE_GUIDE}
{CITATION_FORMAT}
{QUALITY_GUIDELINES}
{COMMON_GUIDELINES}
"""

    
proofreader_agent = Agent(
    name="ProofreaderAgent",
    instructions=INSTRUCTIONS,
    model=reasoning_model
)