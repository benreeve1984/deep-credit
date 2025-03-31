import asyncio
import time
from .iterative_research import IterativeResearcher
from .agents.planner_agent import planner_agent, ReportPlan, ReportPlanSection
from .agents.proofreader_agent import ReportDraftSection, ReportDraft, proofreader_agent
from typing import List, Optional
from agents import Runner
from agents.tracing import trace, gen_trace_id, custom_span
from .logger import DeepCreditLogger

class DeepResearcher:
    """
    Manager for the deep research workflow that breaks down a query into a report plan with sections and then runs an iterative research loop for each section.
    """
    def __init__(
            self, 
            max_iterations: int = 5,
            max_time_minutes: int = 10,
            verbose: bool = True,
            tracing: bool = False,
            logger: Optional[DeepCreditLogger] = None,
            company_name: str = "Unknown Company"
        ):
        self.max_iterations = max_iterations
        self.max_time_minutes = max_time_minutes
        self.verbose = verbose
        self.tracing = tracing
        
        # Set up logger if not provided
        if logger is None and verbose:
            self.logger = DeepCreditLogger(company_name=company_name, mode="deep")
        else:
            self.logger = logger

        if not self.tracing:
            from agents import set_tracing_disabled
            set_tracing_disabled(True)

    async def run(self, query: str) -> str:
        """Run the deep research workflow"""
        start_time = time.time()

        if self.tracing:
            trace_id = gen_trace_id()
            workflow_trace = trace("deep_researcher", trace_id=trace_id)
            if self.logger:
                self.logger.detail(f"Trace ID: {trace_id}")
                self.logger.high_level(f"View trace: https://platform.openai.com/traces/{trace_id}")
            else:
                print(f"View trace: https://platform.openai.com/traces/{trace_id}")
            workflow_trace.start(mark_as_current=True)

        if self.logger:
            self.logger.section_break("Starting Deep Research Workflow")
            self.logger.detail(f"Query: {query}")
            self.logger.detail(f"Max iterations per section: {self.max_iterations}")
            self.logger.detail(f"Max time: {self.max_time_minutes} minutes")
        
        # First build the report plan which outlines the sections and compiles any relevant background context on the query
        report_plan = await self._build_report_plan(query)

        # Run the independent research loops concurrently for each section and gather the results
        research_results = await self._run_research_loops(report_plan)

        # Create the final report from the original report plan and the drafts of each section
        final_report = await self._create_final_report(query, report_plan, research_results)

        elapsed_time = int(time.time() - start_time)
        if self.logger:
            self.logger.research_complete(len(report_plan.report_outline), elapsed_time)
        else:
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            print(f"DeepResearcher completed in {minutes} minutes and {seconds} seconds")

        if self.tracing:
            workflow_trace.finish(reset_current=True)

        return final_report

    async def _build_report_plan(self, query: str) -> ReportPlan:
        """Build the initial report plan including the report outline (sections and key questions) and background context"""
        if self.tracing:
            span = custom_span(name="build_report_plan")
            span.start(mark_as_current=True)

        if self.logger:
            self.logger.section_break("Building Report Plan")
        else:
            self._log_message("=== Building Report Plan ===")
            
        user_message = f"QUERY: {query}"
        
        if self.logger:
            self.logger.detail(f"Query for planner agent: {user_message}")
            self.logger.api_call()
        
        result = await Runner.run(
            planner_agent,
            user_message
        )
        report_plan = result.final_output_as(ReportPlan)

        num_sections = len(report_plan.report_outline)
        if self.logger:
            self.logger.high_level(f"Report plan created with {num_sections} sections:")
            for i, section in enumerate(report_plan.report_outline):
                self.logger.high_level(f"{i+1}. {section.title}")
                self.logger.detail(f"Section {i+1} key question: {section.key_question}")
            
            if report_plan.background_context:
                self.logger.detail(f"Background context: {report_plan.background_context[:200]}...")
            else:
                self.logger.detail("No background context was provided.")
        else:
            message_log = '\n\n'.join(f"Section: {section.title}\nKey question: {section.key_question}" for section in report_plan.report_outline)
            if report_plan.background_context:
                message_log += f"\n\nThe following background context has been included for the report build:\n{report_plan.background_context}"
            else:
                message_log += "\n\nNo background context was provided for the report build.\n"
            self._log_message(f"Report plan created with {num_sections} sections:\n{message_log}")

        if self.tracing:
            span.finish(reset_current=True)

        return report_plan

    async def _run_research_loops(
        self, 
        report_plan: ReportPlan
    ) -> List[str]:
        """For a given ReportPlan, run a research loop concurrently for each section and gather the results"""
        async def run_research_for_section(section: ReportPlanSection, section_index: int):
            if self.logger:
                self.logger.high_level(f"\nResearching section {section_index+1}: {section.title}")
                section_logger = None  # We'll use the same logger for all sections
            else:
                section_logger = None
                
            iterative_researcher = IterativeResearcher(
                max_iterations=self.max_iterations,
                max_time_minutes=self.max_time_minutes,
                verbose=self.verbose,
                tracing=False,  # Do not trace as this will conflict with the tracing we already have set up for the deep researcher
                logger=section_logger
            )
            args = {
                "query": section.key_question,
                "output_length": "",
                "output_instructions": "",
                "background_context": report_plan.background_context,
            }
            
            if self.logger:
                self.logger.detail(f"Starting research for section: {section.title}")
                self.logger.detail(f"Key question: {section.key_question}")
            
            # Only use custom span if tracing is enabled
            if self.tracing:
                with custom_span(
                    name=f"iterative_researcher:{section.title}", 
                    data={"key_question": section.key_question}
                ):
                    result = await iterative_researcher.run(**args)
            else:
                result = await iterative_researcher.run(**args)
                
            if self.logger:
                self.logger.detail(f"Completed research for section: {section.title}")
                self.logger.detail(f"Result length: {len(result)} characters")
                
            return result
        
        if self.logger:
            self.logger.section_break("Running Research for Each Section")
            self.logger.high_level(f"Processing {len(report_plan.report_outline)} sections in parallel")
        else:
            self._log_message("=== Initializing Research Loops ===")
            
        # Run all research loops concurrently in a single gather call
        tasks = []
        for i, section in enumerate(report_plan.report_outline):
            tasks.append(run_research_for_section(section, i))
            
        research_results = await asyncio.gather(*tasks)
        
        if self.logger:
            self.logger.high_level(f"Completed research for all {len(report_plan.report_outline)} sections")
            
        return research_results

    async def _create_final_report(
        self, 
        query: str, 
        report_plan: ReportPlan, 
        section_drafts: List[str]
    ) -> str:
        """Create the final report from the original report plan and the drafts of each section"""
        if self.tracing:
            span = custom_span(name="create_final_report")
            span.start(mark_as_current=True)

        # Each section is a string containing the markdown for the section
        # From this we need to build a ReportDraft object to feed to the final proofreader agent
        report_draft = ReportDraft(
            sections=[]
        )
        for i, section_draft in enumerate(section_drafts):
            report_draft.sections.append(
                ReportDraftSection(
                    section_title=report_plan.report_outline[i].title,
                    section_content=section_draft
                )
            )

        user_prompt = f"QUERY:\n{query}\n\nREPORT DRAFT:\n{report_draft.model_dump_json()}"

        if self.logger:
            self.logger.section_break("Building Final Report")
            self.logger.detail(f"Assembling {len(report_draft.sections)} sections")
            for i, section in enumerate(report_draft.sections):
                self.logger.detail(f"Section {i+1}: {section.section_title} - {len(section.section_content)} characters")
        else:
            self._log_message("\n=== Building Final Report ===")
            
        # Run the proofreader agent to produce the final report
        if self.logger:
            self.logger.api_call()
            
        final_report = await Runner.run(
            proofreader_agent,
            user_prompt
        )
        
        if self.logger:
            self.logger.high_level("Final report completed")
            self.logger.detail(f"Final report length: {len(final_report.final_output)} characters")
        else:
            self._log_message(f"Final report completed")

        if self.tracing:
            span.finish(reset_current=True)

        return final_report.final_output

    def _log_message(self, message: str) -> None:
        """Legacy method for backward compatibility - use logger instead"""
        if self.logger:
            if message.startswith("==="):
                # Section header
                self.logger.detail(f"Legacy log message: {message}")
            else:
                # Other information
                self.logger.detail(f"Legacy log message: {message}")
        elif self.verbose:
            print(message)