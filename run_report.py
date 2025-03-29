"""
A flexible script to run Deep Credit rating reports with configurable parameters.
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path
import logging
import sys
from deep_researcher import DeepResearcher, IterativeResearcher

# Configuration parameters
CONFIG = {
    # Common parameters
    "max_time_minutes": 15,  # Increased for more thorough analysis
    "verbose": True,
    "tracing": True,
    
    # Research mode
    "default_mode": "deep",  # Set to "deep" or "iterative"
    
    # Deep research specific
    "deep_max_iterations": 5,  # Increased for more thorough analysis
    
    # Iterative research specific
    "iterative_max_iterations": 5,
    "output_length": "10 pages",  # Increased for comprehensive report
    "output_instructions": "",
    
    # Company to analyze
    "company_name": "Trafigura Group Pte. Ltd.",  # Hard coded company name
    
    # Default query template
    "query_template": '''You are a Deep Research agentic LLM with extensive expertise in credit rating methodologies. Your objective is to produce a single, final credit rating for {company_name}, specifically covering its senior unsecured long-term debt instruments (e.g., bonds or notes issued without collateral), using only publicly accessible data and best practice frameworks from recognized rating agencies (e.g., Moody's, S&P, Fitch). Follow these steps to arrive at your final rating:

1. Outline Methodology & Rationale
   • Summarize standard credit-rating methodologies from major agencies, emphasizing factors such as macroeconomic conditions, industry outlook, financial performance, capital structure, liquidity, and corporate governance.
   • Describe how each factor influences the credit rating process for senior unsecured debt.

2. Data Gathering & Analysis
   • Identify and compile all relevant data from freely available sources (e.g., annual reports, reputable news outlets, financial aggregators).
   • List these data sources, specifying the key information each provides (e.g., financial ratios, revenue growth, debt levels).
   • Calculate or reference important financial metrics (e.g., debt-to-EBITDA, interest coverage, profitability) based on the public data gathered.

3. Risk & Creditworthiness Assessment
   • Evaluate business risk factors (e.g., competitive dynamics, regulatory environment, market trends).
   • Assess financial risk by examining leverage, cash flow stability, and liquidity management.
   • Integrate qualitative considerations (e.g., management track record, corporate governance).
   • Where possible, compare metrics to industry peers to contextualize risk.

4. Information Gaps & Confidence
   • Explicitly note any material data you could not obtain (e.g., recent financial statements, insider commentary).
   • Discuss how these gaps might impact the precision of your analysis.
   • Assign and explain your level of confidence (high/medium/low) in the final assessment, given these limitations.

5. Arrive at a Single Credit Rating
   • Based on your analysis, assign one final rating (e.g., AAA, AA, A, BBB, etc.) specifically for the senior unsecured long-term debt.
   • Justify this rating by referencing the major strengths, weaknesses, and any external factors.

6. Detailed Written Report
   • Present a structured credit rating report incorporating your methodology, data findings, and rationale.
   • Highlight both quantitative and qualitative reasoning.
   • Conclude with a clear statement of the single, final rating for the senior unsecured long-term debt instruments and clarify the confidence level in your conclusion.

Ensure your report is self-contained, thoroughly explaining how you arrived at your final rating based on publicly available information.''',
            
    # Output settings
    "output_dir": "credit_ratings"  # Changed directory name
}

class ConsoleHandler(logging.Handler):
    """Custom logging handler that only writes to console."""
    def __init__(self):
        super().__init__()
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.api_call_count = 0

    def emit(self, record):
        try:
            msg = self.format(record)
            # Skip HTTP request logs
            if "HTTP Request:" in msg:
                self.api_call_count += 1
                return
            # Add API call summary at the end
            if "Report saved to:" in msg:
                msg += f"\n\nTotal API calls made: {self.api_call_count}"
            self.console_handler.emit(record)
        except Exception:
            self.handleError(record)

def setup_logging():
    """Set up logging to write only to console."""
    # Set up custom handler that writes only to console
    handler = ConsoleHandler()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

def save_report(report: str, output_path: Path):
    """Save the report to a file."""
    # Create output directory if it doesn't exist
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True)
    
    # Write report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

async def run_deep_research(query, max_iterations, max_time_minutes, verbose, tracing):
    """Run a deep credit rating report."""
    manager = DeepResearcher(
        max_iterations=max_iterations,
        max_time_minutes=max_time_minutes,
        verbose=verbose,
        tracing=tracing
    )
    
    report = await manager.run(query)
    return report

async def run_iterative_research(query, max_iterations, max_time_minutes, verbose, tracing,
                               output_length, output_instructions):
    """Run an iterative credit rating report."""
    manager = IterativeResearcher(
        max_iterations=max_iterations,
        max_time_minutes=max_time_minutes,
        verbose=verbose,
        tracing=tracing
    )
    
    report = await manager.run(
        query,
        output_length=output_length,
        output_instructions=output_instructions
    )
    return report

def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be used as a filename."""
    # Replace newlines and multiple spaces with single space
    filename = ' '.join(filename.split())
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Limit length and remove trailing spaces/dots
    filename = filename[:100].strip('. ')
    return filename

def main():
    parser = argparse.ArgumentParser(description='Run a credit rating report using either deep or iterative approach')
    parser.add_argument('--mode', choices=['deep', 'iterative'], default=CONFIG['default_mode'],
                      help='Research mode to use (default: from CONFIG)')
    parser.add_argument('--max-time', type=int, default=CONFIG['max_time_minutes'],
                      help='Maximum time in minutes (default: 15)')
    parser.add_argument('--verbose', action='store_true', default=CONFIG['verbose'],
                      help='Enable verbose output')
    parser.add_argument('--tracing', action='store_true', default=CONFIG['tracing'],
                      help='Enable tracing')
    parser.add_argument('--output-length', type=str, default=CONFIG['output_length'],
                      help='Desired output length (for iterative mode)')
    parser.add_argument('--output-instructions', type=str, default=CONFIG['output_instructions'],
                      help='Additional output instructions (for iterative mode)')
    
    args = parser.parse_args()
    
    # Set max iterations based on mode
    max_iterations = (CONFIG['deep_max_iterations'] if args.mode == 'deep' 
                     else CONFIG['iterative_max_iterations'])
    
    # Create query using template
    query = CONFIG['query_template'].format(company_name=CONFIG['company_name'])
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    company_snippet = sanitize_filename(CONFIG['company_name'][:50])
    filename = f"{timestamp}_{args.mode}_{company_snippet}_credit_rating.txt"
    
    # Set up logging to console only
    setup_logging()
    
    # Log initial information to console
    logging.info(f"Research Mode: {args.mode}")
    logging.info(f"Company: {CONFIG['company_name']}")
    logging.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("\n" + "="*50 + "\n")
    
    logging.info(f"\nRunning {args.mode} credit rating analysis...")
    logging.info(f"Company: {CONFIG['company_name']}")
    logging.info(f"Max time: {args.max_time} minutes")
    logging.info(f"Max iterations: {max_iterations}")
    
    if args.mode == 'deep':
        report = asyncio.run(run_deep_research(
            query,
            max_iterations,
            args.max_time,
            args.verbose,
            args.tracing
        ))
    else:
        report = asyncio.run(run_iterative_research(
            query,
            max_iterations,
            args.max_time,
            args.verbose,
            args.tracing,
            args.output_length,
            args.output_instructions
        ))
    
    # Save report to file
    output_path = Path(__file__).parent / CONFIG['output_dir'] / filename
    save_report(report, output_path)
    
    # Log completion to console
    logging.info(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    main() 