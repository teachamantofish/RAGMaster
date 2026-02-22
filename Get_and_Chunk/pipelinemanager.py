#!/usr/bin/env python3
"""
Pipeline Manager - Orchestrates the complete RAG pipeline

Executes the numbered pipeline scripts in order:
1. Crawler (selected based on PARSER column from CSV)
2. Markdown cleanup
3. Chunking
4. Summarization
5. Embedding
6. Vector database upload

The crawler selection is based on the PARSER field in metadataconfig.csv:
- crawlweb -> 1crawler_web.py
- crawlpdf -> 1crawler_pdf.py
- crawlgit -> 1crawler_github.py
"""

import subprocess
import sys
import os
from pathlib import Path
from common.utils import (get_csv_to_process, setup_global_logger)

# Get CSV configuration
csvrow_data = get_csv_to_process()
metadata = csvrow_data['input_csv_row']
CWD = csvrow_data['cwd']

# Set up global logger
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Script", "Exit Code"]
logger = setup_global_logger(script_name=script_base, log_level='INFO', headers=LOG_HEADER)

# Mapping from PARSER column to crawler script
CRAWLER_MAPPING = {
    'crawlweb': '1crawler_web.py',
    'crawlpdf': '1crawler_pdf.py', 
    'crawlgit': '1crawler_github.py'
}

# Pipeline steps (after crawler selection)
PIPELINE_STEPS = [
    '2markdown_cleanup.py',
    '3chunker.py',
    '4summary.py',
    '5embedding.py',
    '6vector.py'
]

def run_script(script_name: str, step_number: int = None) -> int:
    """
    Run a Python script and return its exit code.
    
    Args:
        script_name: Name of the Python script to run
        step_number: Optional step number for logging context
        
    Returns:
        Exit code from the script (0 = success, non-zero = error)
    """
    step_info = f" (Step {step_number})" if step_number else ""
    logger.info(f"Starting {script_name}{step_info}", extra={"Script": script_name, "Exit Code": ""})
    
    try:
        # Run the script using subprocess
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path.cwd(),  # Run in current directory
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        exit_code = result.returncode
        
        if exit_code == 0:
            logger.info(f"Completed {script_name}{step_info} successfully", 
                       extra={"Script": script_name, "Exit Code": exit_code})
        else:
            logger.error(f"Failed {script_name}{step_info} with exit code {exit_code}", 
                        extra={"Script": script_name, "Exit Code": exit_code})
            if result.stderr:
                logger.error(f"Error output: {result.stderr.strip()}")
                
        return exit_code
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout running {script_name}{step_info}", 
                    extra={"Script": script_name, "Exit Code": "TIMEOUT"})
        return -1
        
    except Exception as e:
        logger.error(f"Exception running {script_name}{step_info}: {e}", 
                    extra={"Script": script_name, "Exit Code": "EXCEPTION"})
        return -1

def run_pipeline() -> bool:
    """
    Execute the complete RAG pipeline.
    
    Returns:
        True if all steps completed successfully, False if any step failed
    """
    # Get parser type from CSV metadata
    parser_type = metadata.get('PARSER', '').strip().lower()
    crawl_id = metadata.get('ID', 'unknown')
    
    logger.info(f"Starting RAG pipeline for crawl ID {crawl_id} with parser type '{parser_type}'")
    
    # Step 1: Select and run appropriate crawler
    if parser_type not in CRAWLER_MAPPING:
        logger.error(f"Unknown parser type '{parser_type}'. Valid types: {list(CRAWLER_MAPPING.keys())}")
        return False
        
    crawler_script = CRAWLER_MAPPING[parser_type]
    logger.info(f"Selected crawler: {crawler_script} for parser type '{parser_type}'")
    
    exit_code = run_script(crawler_script, step_number=1)
    if exit_code != 0:
        logger.error(f"Pipeline failed at Step 1 ({crawler_script})")
        return False
    
    # Steps 2-6: Run remaining pipeline steps
    for i, script in enumerate(PIPELINE_STEPS, start=2):
        exit_code = run_script(script, step_number=i)
        if exit_code != 0:
            logger.error(f"Pipeline failed at Step {i} ({script})")
            return False
    
    logger.info("RAG pipeline completed successfully!")
    return True

def main():
    """Main entry point for pipeline execution."""
    logger.info(f"Pipeline Manager starting for crawl ID {metadata.get('ID', 'unknown')}")
    logger.info(f"Working directory: {CWD}")
    logger.info(f"Target URL: {metadata.get('CRAWL_URL', 'unknown')}")
    
    success = run_pipeline()
    
    if success:
        logger.info("Pipeline execution completed successfully")
        sys.exit(0)
    else:
        logger.error("Pipeline execution failed")
        sys.exit(1)

if __name__ == '__main__':
    main()