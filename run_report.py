"""
A flexible script to run Deep Credit rating reports with configurable parameters.
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path
import sys
from deep_researcher import DeepResearcher, IterativeResearcher
from deep_researcher.logger import DeepCreditLogger

# Configuration parameters
CONFIG = {
    # Common parameters
    "max_time_minutes": 30,  # Increased for more thorough analysis
    "verbose": True,
    "tracing": True,
    
    # Research mode
    "default_mode": "deep",  # Set to "deep" or "iterative"
    
    # Deep research specific
    "deep_max_iterations": 8,  # Increased for more thorough analysis
    
    # Iterative research specific
    "iterative_max_iterations": 8,
    "output_length": "comprehensive (at least 10 pages)",  # Increased for detailed report
    "output_instructions": "Provide detailed financial analysis with specific metrics and values. Use full paragraphs rather than bullet points.",
    
    # Company to analyze
    "company_name": "Trafigura Group Pte. Ltd.",  # Hard coded company name
    
    # Default query template
    "query_template": '''Prepare a comprehensive, detailed credit rating report (minimum 10 pages) for {company_name}'s senior unsecured long-term debt. 

The final report must include:
1. A single, final letter-grade credit rating (e.g., AAA, AA+, BBB-, Ba1)
2. A rating outlook (Stable, Positive, or Negative)
3. Thorough analysis based on publicly available information
4. Specific financial metrics with actual values
5. Multi-year trends and comparative analysis
6. Peer company comparisons
7. Clear explanation of the rating rationale

Focus directly on analyzing {company_name} rather than explaining rating methodologies.''',
            
    # Output settings
    "output_dir": "credit_ratings",  # Changed directory name
    "log_dir": "logs"  # Directory for detailed logs
}

def save_report(report: str, output_path: Path):
    """Save the report to a file."""
    # Create output directory if it doesn't exist
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True)
    
    # Write report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

async def run_deep_research(query, max_iterations, max_time_minutes, verbose, tracing, logger):
    """Run a deep credit rating report."""
    manager = DeepResearcher(
        max_iterations=max_iterations,
        max_time_minutes=max_time_minutes,
        verbose=verbose,
        tracing=tracing,
        logger=logger,
        company_name=CONFIG['company_name']
    )
    
    report = await manager.run(query)
    return report

async def run_iterative_research(query, max_iterations, max_time_minutes, verbose, tracing,
                               output_length, output_instructions, logger):
    """Run an iterative credit rating report."""
    manager = IterativeResearcher(
        max_iterations=max_iterations,
        max_time_minutes=max_time_minutes,
        verbose=verbose,
        tracing=tracing,
        logger=logger,
        company_name=CONFIG['company_name']
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
                      help='Maximum time in minutes (default: 30)')
    parser.add_argument('--verbose', action='store_true', default=CONFIG['verbose'],
                      help='Enable verbose output')
    parser.add_argument('--tracing', action='store_true', default=CONFIG['tracing'],
                      help='Enable tracing')
    parser.add_argument('--output-length', type=str, default=CONFIG['output_length'],
                      help='Desired output length (for iterative mode)')
    parser.add_argument('--output-instructions', type=str, default=CONFIG['output_instructions'],
                      help='Additional output instructions (for iterative mode)')
    parser.add_argument('--log-dir', type=str, default=CONFIG['log_dir'],
                      help='Directory for detailed logs')
    
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
    
    # Set up logger
    logger = DeepCreditLogger(
        company_name=CONFIG['company_name'],
        mode=args.mode,
        log_dir=args.log_dir
    )
    
    # Log initial information
    logger.section_break("Credit Rating Analysis")
    logger.high_level(f"Company: {CONFIG['company_name']}")
    logger.high_level(f"Research Mode: {args.mode}")
    logger.high_level(f"Max time: {args.max_time} minutes | Max iterations: {max_iterations}")
    logger.high_level(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log detailed configuration
    logger.detail(f"Full Query: {query}")
    logger.detail(f"Output Length: {args.output_length}")
    logger.detail(f"Output Instructions: {args.output_instructions}")
    logger.detail(f"Log directory: {args.log_dir}")
    logger.detail(f"Verbose: {args.verbose} | Tracing: {args.tracing}")
    
    # Run the appropriate research method
    if args.mode == 'deep':
        logger.section_break("Starting Deep Research")
        report = asyncio.run(run_deep_research(
            query,
            max_iterations,
            args.max_time,
            args.verbose,
            args.tracing,
            logger
        ))
    else:
        logger.section_break("Starting Iterative Research")
        report = asyncio.run(run_iterative_research(
            query,
            max_iterations,
            args.max_time,
            args.verbose,
            args.tracing,
            args.output_length,
            args.output_instructions,
            logger
        ))
    
    # Save report to file
    output_path = Path(__file__).parent / CONFIG['output_dir'] / filename
    save_report(report, output_path)
    
    # Log completion info
    logger.report_saved(output_path)
    logger.high_level(f"Detailed log saved to: {logger.get_log_file_path()}")
    
if __name__ == "__main__":
    main() 