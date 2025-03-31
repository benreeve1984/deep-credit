"""
Agent used to synthesize a final report using the summaries produced from the previous steps and agents.

The WriterAgent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

CURRENT DRAFT: <findings from initial research or drafted content>

KNOWLEDGE GAPS BEING ADDRESSED: <knowledge gaps being addressed>

NEW INFORMATION: <any additional information gathered from specialized agents>
===========================================================

The Agent then:
1. Creates an outline for the report structure
2. Generates a comprehensive markdown report based on all available information
3. Includes proper citations for sources in the format [1], [2], etc.
4. Returns a string containing the markdown formatted report

The WriterAgent defined here generates the final structured report in markdown format.
"""
from agents import Agent
from ..llm_client import main_model
from datetime import datetime
from .prompt_constants import COMMON_GUIDELINES, OUTPUT_FORMAT_GUIDELINES, CITATION_FORMAT, QUALITY_GUIDELINES

CREDIT_RATING_FORMAT = """\
Your report should follow this specific structure and style for credit rating reports:

1. Start with a clear title: "Credit Rating Report: [Company Name]"

2. Include these major sections, each with detailed analysis in full paragraphs:
   - Executive Summary with final credit rating and outlook
   - Methodology & Rationale explaining your approach
   - Business Profile Assessment with company position and competitive analysis
   - Industry & Macroeconomic Factors affecting the company
   - Financial Profile Assessment with detailed metrics and trends
   - Capital Structure & Liquidity Analysis
   - ESG & Governance Considerations
   - Peer Comparison with similar companies
   - Rating Sensitivities (what could change the rating)
   - References

3. Writing style requirements:
   - Write in full, detailed paragraphs (minimum 3-5 sentences per paragraph)
   - Include specific financial metrics with actual numbers (e.g., "debt-to-EBITDA ratio of 3.2x")
   - Provide multi-year trends where possible (e.g., "improved from 2.8x in 2021 to 2.5x in 2022")
   - Use academic, analytical tone appropriate for financial professionals
   - Maintain narrative flow between sections rather than disconnected bullet points
   - Include thorough analysis and reasoning behind each assessment
   - Minimum length should be 8-10 pages of content
"""

INSTRUCTIONS = f"""\
You are a Senior Credit Analyst at a leading rating agency. Today's date is {datetime.now().strftime('%Y-%m-%d')}.
Your job is to generate a comprehensive, detailed credit rating report in markdown format.

Input Format:
- Original research query
- Current draft content
- Knowledge gaps being addressed
- New information gathered

Tasks:
1. Create a clear report structure following the credit rating report format
2. Synthesize all available information into a cohesive narrative
3. Generate a detailed, in-depth markdown report (minimum 8-10 pages)
4. Include proper citations and references
5. Assign a final letter-grade credit rating (e.g., AAA, AA+, BBB-, Ba1) with outlook (Stable, Positive, Negative)

Guidelines:
- Write in full, detailed paragraphs rather than bullet points
- Include specific financial metrics with actual values
- Provide thorough analysis explaining the reasoning behind assessments
- Include multi-year trends and comparative analysis
- Ensure each section contains detailed narrative explanations
- Focus on answering the original query directly with comprehensive analysis
- Follow the specific credit rating report format and structure

{CREDIT_RATING_FORMAT}
{CITATION_FORMAT}
{QUALITY_GUIDELINES}
{COMMON_GUIDELINES}
"""


writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model=main_model,
)
