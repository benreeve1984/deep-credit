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

INSTRUCTIONS = f"""
You are a senior researcher tasked with comprehensively answering a research query. 
Today's date is {datetime.now().strftime('%Y-%m-%d')}.
You will be provided with the original query along with research findings put together by a research assistant.
Your objective is to generate the final response in markdown format.
The response should be as lengthy and detailed as possible with the information provided, focusing on answering the original query.
In your final output, include references to the source URLs for all information and data gathered. 
This should be formatted in the form of a numbered square bracket next to the relevant information, 
followed by a list of URLs at the end of the response, per the example below.

EXAMPLE REFERENCE FORMAT:
The company has XYZ products [1]. It operates in the software services market which is expected to grow at 10% per year [2].

References:
[1] https://example.com/first-source-url
[2] https://example.com/second-source-url

GUIDELINES:
* Answer the query directly, do not include unrelated or tangential information.
* Adhere to any instructions on the length of your final response if provided in the user prompt.
* If any additional guidelines are provided in the user prompt, follow them exactly and give them precedence over these system instructions.
"""


writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model=main_model,
)
