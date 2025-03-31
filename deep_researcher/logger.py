"""
Logger module for DeepCredit to manage both high-level progress indicators and detailed logging.
This module provides two types of logging:
1. Console logging: Shows high-level progress information for the user to follow
2. File logging: Captures detailed information for later analysis
"""

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path


class LogLevel(Enum):
    """Log levels for different types of messages"""
    HIGH_LEVEL = 25  # Between INFO and WARNING
    PROGRESS = 15  # Between DEBUG and INFO
    DETAIL = 10  # Same as DEBUG


class DeepCreditLogger:
    """Logger for DeepCredit that handles both console and file logging"""
    
    def __init__(self, company_name: str, mode: str, log_dir: str = "logs"):
        """
        Initialize the logger
        
        Args:
            company_name: Name of the company being analyzed
            mode: Research mode (deep or iterative)
            log_dir: Directory to store log files
        """
        self.company_name = company_name
        self.mode = mode
        self.log_dir = log_dir
        self.api_call_count = 0
        
        # Add custom log level
        logging.addLevelName(LogLevel.HIGH_LEVEL.value, "HIGH_LEVEL")
        logging.addLevelName(LogLevel.PROGRESS.value, "PROGRESS")
        
        # Create main logger
        self.logger = logging.getLogger("deepcredit")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Set up console handler for high-level messages
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(LogLevel.HIGH_LEVEL.value)
        self.console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.console_handler)
        
        # Set up file handler for detailed logging
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_company = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in company_name)
        sanitized_company = sanitized_company[:50].strip()
        self.log_file = Path(log_dir) / f"{timestamp}_{mode}_{sanitized_company}.log"
        
        self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        self.file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(file_formatter)
        self.logger.addHandler(self.file_handler)
        
        self.start_time = datetime.now()

    def high_level(self, message: str) -> None:
        """Log high-level information (visible in console)"""
        self.logger.log(LogLevel.HIGH_LEVEL.value, message)
        
    def progress(self, message: str) -> None:
        """Log progress information (file only)"""
        self.logger.log(LogLevel.PROGRESS.value, message)
        
    def detail(self, message: str) -> None:
        """Log detailed information (file only)"""
        self.logger.debug(message)
        
    def api_call(self) -> None:
        """Track API call"""
        self.api_call_count += 1
        self.detail(f"API call made (total: {self.api_call_count})")
        
    def error(self, message: str) -> None:
        """Log error messages (both console and file)"""
        self.logger.error(message)
    
    def warning(self, message: str) -> None:
        """Log warning messages (both console and file)"""
        self.logger.warning(message)
        
    def section_break(self, section_name: str = "") -> None:
        """Create a section break in the logs"""
        if section_name:
            header = f"=== {section_name} ==="
        else:
            header = "=" * 50
        
        self.high_level(f"\n{header}")
        self.detail(f"\n{header}")
        
    def iteration_start(self, iteration_number: int, max_iterations: int) -> None:
        """Log the start of an iteration"""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        progress = f"Iteration {iteration_number}/{max_iterations} ({elapsed:.1f} min elapsed)"
        self.high_level(f"\n=== {progress} ===")
        self.detail(f"Starting iteration {iteration_number} of {max_iterations}")
        
    def iteration_finish(self, iteration_number: int) -> None:
        """Log the completion of an iteration"""
        self.detail(f"Completed iteration {iteration_number}")
        
    def knowledge_gap(self, gap: str, priority: int, attempted: bool) -> None:
        """Log a knowledge gap"""
        status = "Previously Attempted" if attempted else "New Gap"
        short_gap = gap[:80] + "..." if len(gap) > 80 else gap
        
        # High-level shows truncated version
        self.high_level(f"▶ Gap: {short_gap}")
        self.high_level(f"  Priority: {priority}/5 | {status}")
        
        # Detailed log shows full text
        self.detail(f"Knowledge Gap: {gap}")
        self.detail(f"Gap Priority: {priority}/5 | Previously Attempted: {attempted}")
        
    def tool_execution(self, agent_name: str, query: str) -> None:
        """Log a tool execution"""
        short_query = query[:50] + "..." if len(query) > 50 else query
        
        # High-level shows only agent and shortened query
        self.high_level(f"  ◆ Using {agent_name}: \"{short_query}\"")
        
        # Detailed logs show full information
        self.detail(f"Executing Tool: {agent_name}")
        self.detail(f"Tool Query: {query}")
        
    def tool_result(self, agent_name: str, result_length: int) -> None:
        """Log the result of a tool execution"""
        self.detail(f"Tool Result from {agent_name}: {result_length} characters received")
        
    def research_complete(self, iterations: int, elapsed_seconds: int) -> None:
        """Log completion of research"""
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        self.high_level(f"\n=== Research Complete ===")
        self.high_level(f"Completed {iterations} iterations in {minutes}m {seconds}s")
        self.high_level(f"Total API calls: {self.api_call_count}")
        self.detail(f"Research process completed after {iterations} iterations")
        self.detail(f"Total elapsed time: {minutes}m {seconds}s")
        self.detail(f"Total API calls: {self.api_call_count}")
        
    def report_saved(self, output_path: Path) -> None:
        """Log that the report has been saved"""
        self.high_level(f"\nReport saved to: {output_path}")
        self.detail(f"Final report saved to: {output_path}")
        
    def get_log_file_path(self) -> Path:
        """Get the path to the log file"""
        return self.log_file 