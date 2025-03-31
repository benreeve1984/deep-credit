"""
Agent used to evaluate the state of the research report (typically done in a loop) and identify knowledge gaps that still 
need to be addressed.

The Agent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

CURRENT DRAFT: <most recent draft of the research output>

PREVIOUS EVALUATION: <the KnowledgeGapOutput from the previous iteration>
===========================================================

The Agent then:
1. Carefully reviews the current draft and assesses its completeness in answering the original query
2. Identifies specific knowledge gaps that still exist and need to be filled
3. Returns a KnowledgeGapOutput object
"""

from pydantic import BaseModel, Field
from typing import List
from agents import Agent
from ..llm_client import fast_model
from datetime import datetime
from .prompt_constants import COMMON_GUIDELINES, QUALITY_GUIDELINES

class KnowledgeGapOutput(BaseModel):
    """Output from the Knowledge Gap Agent"""
    research_complete: bool = Field(description="Whether the research and findings are complete enough to end the research loop")
    outstanding_gaps: List[str] = Field(description="List of knowledge gaps that still need to be addressed")
    gap_priorities: List[int] = Field(description="Priority levels for each gap (1=highest, 5=lowest)")
    previously_attempted: List[bool] = Field(description="Whether each gap has been attempted in previous iterations")


CREDIT_RATING_REQUIRED_ELEMENTS = """\
For a comprehensive credit rating report, ensure these critical elements are present before marking research as complete:

1. Financial Metrics (must have actual numerical values):
   - Debt-to-EBITDA ratio
   - Interest coverage ratio
   - Free cash flow metrics
   - Liquidity ratios
   - Revenue/EBITDA/profit margins trends (multi-year)
   - Capital structure breakdown

2. Business Risk Elements:
   - Market position with specific market share data
   - Detailed competitive landscape
   - Geographic diversification
   - Business model sustainability analysis

3. Industry Analysis:
   - Industry growth rates and trends
   - Regulatory environment details
   - Comparison to industry averages

4. Capital Structure:
   - Detailed debt maturity schedule
   - Refinancing risks
   - Covenant details

5. Peer Comparison:
   - Side-by-side comparison with at least 2-3 peers
   - Key metrics benchmarking

However, if after multiple attempts certain information proves unobtainable, the research can proceed with other important elements.
"""


INSTRUCTIONS = f"""\
You are a Credit Rating Research Manager. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
Your job is to analyze the current state of a credit rating report, identify knowledge gaps, and ensure the research covers the full scope even when some specific information is difficult to obtain.

Input Format:
1. Original user query and background context
2. Current draft content
3. Previous evaluation (if available)
4. Full history of tasks, actions, findings, and thoughts

Tasks:
1. Review the draft credit rating report and assess its completeness
2. Identify knowledge gaps that still need to be addressed, considering what's feasible to obtain
3. Prioritize gaps based on importance and feasibility
4. Track which gaps have been attempted in previous iterations
5. Suggest alternative approaches for persistent gaps that have been attempted multiple times
6. Decide when research is sufficiently complete to move forward

Adaptive Research Guidelines:
- Don't get stuck on the same gaps repeatedly - after 2 attempts, lower their priority or suggest alternatives
- Balance depth vs. breadth - ensure coverage of all major rating areas rather than perfect information in some areas
- If specific numerical data is unavailable after multiple attempts, accept qualitative assessments with explanation
- Consider information difficulty and adjust requirements accordingly
- Focus on high-impact areas first, but ensure overall coverage of the full rating methodology

Knowledge Gap Adaptation:
- For persistent gaps, suggest broader search terms, alternative metrics, or proxy data
- Identify when a gap is truly critical vs. when it would be "nice to have"
- When specific metrics are unavailable, suggest industry benchmarks or reasonable estimates
- Recommend moving forward when sufficient information exists for a reasoned rating judgment

Output Format:
- research_complete: bool - Whether research has sufficient information for a rating decision
- outstanding_gaps: List[str] - List of knowledge gaps to address
- gap_priorities: List[int] - Priority level for each gap (1=highest, 5=lowest)
- previously_attempted: List[bool] - Whether each gap has been attempted before

IMPORTANT: You MUST ALWAYS include values for gap_priorities and previously_attempted, with one value for each gap in outstanding_gaps. 
For gap_priorities, use 1 for high priority items and 5 for low priority. For previously_attempted, check the previous evaluation to see 
which gaps have been attempted before and mark them as true, otherwise use false for newly identified gaps.

{CREDIT_RATING_REQUIRED_ELEMENTS}
{QUALITY_GUIDELINES}
{COMMON_GUIDELINES}
"""


knowledge_gap_agent = Agent(
    name="KnowledgeGapAgent",
    instructions=INSTRUCTIONS,
    model=fast_model,
    output_type=KnowledgeGapOutput,
)