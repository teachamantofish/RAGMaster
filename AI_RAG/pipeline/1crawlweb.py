# This crawler code reads the crawlconfig.py file and fetches webpages which meet the requirements.
# The crawlconfig also supplies the crawlid which is an index to the metadataconfig.csv file 
# containing one row for each document to crawl. That file contains the crawlid, the url, 
# and the other items needed which will be stored as metadata later.
import os  # For directory and file operations
import re
from pathlib import Path
import logging  # For logging
import traceback  # For detailed error information
import asyncio  # For async crawling
from urllib.parse import urlparse, urldefrag  # Only urldefrag still used in save_markdown
from langdetect import detect, LangDetectException
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import BrowserConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

# Import static config values
from config.crawlconfig import *
# Shared metadata utilities (centralized CSV loading + merge rules)
from common.metadata_utils import merge_page_metadata
from common.utils import (get_csv_to_process, setup_global_logger)

csvrow_data = get_csv_to_process() # Get the entire csv row to process, based dir, url, user metadata, etc. 
metadata = csvrow_data['input_csv_row'] # Store the row data in a var
CWD: Path = csvrow_data['cwd'] # Extract the rootdir/basedir from the csv row data
CRAWL_URL = metadata['CRAWL_URL']

# Set up global loger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "TBD", "TBD"]
logger = setup_global_logger(script_name=script_base, log_level='INFO', headers=LOG_HEADER)

# --- Save Markdown ---
def save_markdown(markdown_content: str, save_dir: str, url: str):
    """Save markdown applying standardized front matter merge.

    We generate a filename from the URL's last path segment and then prepend
    front matter built from global CSV metadata plus any original page-level
    YAML (if present) merged via merge_page_metadata.
    """
    clean_url = urldefrag(url)[0]
    parsed = urlparse(clean_url)
    last_segment = parsed.path.rstrip('/').split('/')[-1] or 'index'
    stem = os.path.splitext(last_segment)[0] or 'index'
    filename = f"{stem}.md"
    filepath = os.path.join(str(save_dir), filename)

    # Per-page Source URL: override CRAWL_URL so front matter points to the page actually crawled.
    page_meta = dict(metadata)
    page_meta['CRAWL_URL'] = clean_url
    merged_front, cleaned_body = merge_page_metadata(page_meta, markdown_content)
    full_content = merged_front + cleaned_body

    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(full_content)

# Simplified deep crawl function using Crawl4AI's built-in BFSDeepCrawlStrategy
# All configuration values are read directly from crawllconfig.py for consistency
def deep_crawl_urls():
    """
    Execute deep crawling using Crawl4AI's BFS strategy with all settings from config.
    Returns list of crawl results with success/failure status and content.
    """
    async def crawl():
        logger.info("Initializing Crawl4AI deep crawler with BFS strategy...")
        
        # Set up browser configuration - keeping your existing settings
        browser_config = BrowserConfig(headless=True, ignore_https_errors=True)


        # URLPatternFilter - ensure we have a list of patterns
        patterns = URL_PATTERN_FILTERS if isinstance(URL_PATTERN_FILTERS, list) else [URL_PATTERN_FILTERS]
        logger.info(f"URL Pattern Filters: {patterns}")
        filters = [
            URLPatternFilter(patterns=patterns),
        ]

        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=MAX_CRAWL_DEPTH,
            include_external=INCLUDE_EXTERNAL_DOMAIN,
            filter_chain=FilterChain(filters),
        )
        
        #  analyzes text density, link density, HTML structure, and known patterns 
        # (like “nav,” “footer”) to systematically prune extraneous or repetitive sections.
        prune_filter = PruningContentFilter(
            threshold=0.2,
            threshold_type="dynamic",
            min_word_threshold=2
        )

        md_options = {
            "ignore_links": IGNORE_LINKS,
            "ignore_images": IGNORE_IMAGES,
            "escape_html": ESCAPE_HTML,
            "body_width": BODY_WIDTH,
            "skip_internal_links": SKIP_INTERNAL_LINKS,
            "include_sup_sub": INCLUDE_SUP_SUB,
            "heading_style": HEADING_STYLE,
            "list_style": LIST_STYLE,
            "preserve_tables": PRESERVE_TABLES,
            "collapse_whitespace": COLLAPSE_WHITESPACE
        }
        md_generator = DefaultMarkdownGenerator(options=md_options)
        run_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            remove_forms=REMOVE_FORMS,
            remove_overlay_elements=REMOVE_OVERLAY_ELEMENTS,
            excluded_tags=EXCLUDED_TAGS,
            excluded_selector=EXCLUDED_SELECTOR,
            exclude_external_links=EXCLUDE_EXTERNAL_LINKS,
            exclude_social_media_links=EXCLUDE_SOCIAL_MEDIA_LINKS,
            exclude_domains=EXCLUDE_DOMAINS,
            exclude_social_media_domains=EXCLUDE_SOCIAL_MEDIA_DOMAINS,
            deep_crawl_strategy=deep_crawl_strategy # Use BFS deep crawling
            #cache_mode=CacheMode.BYPASS,
            #css_selector=CSS_SELECTOR,
        )
        # Tracking counters only (no in-memory result aggregation)
        total_processed = 0
        saved_count = 0

        # Execute the deep crawl using Crawl4AI's built-in strategy
        logger.info(f"Starting deep crawl from: {CRAWL_URL}")
        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun(CRAWL_URL, config=run_config)
            logger.info(f"Deep crawl completed. Processing {len(results)} results...")

            for result in results:
                total_processed += 1
                url = getattr(result, 'url', None)
                target_domain = urlparse(CRAWL_URL).netloc if url else None
                result_domain = urlparse(url).netloc if url else None
                depth = result.metadata.get('depth', 0) if hasattr(result, 'metadata') else 0
                error_msg = getattr(result, 'error_message', '')

                if url and result_domain != target_domain:
                    logger.info(f"Skipping external domain: {url} (domain: {result_domain} != {target_domain})")
                    continue

                if result.success:
                    md_for_counts = getattr(result, 'markdown', '') or ''
                    num_chars = len(md_for_counts)
                    num_lines_for_log = md_for_counts.count('\n') + 1 if md_for_counts else 0
                    status_field = 'success'
                    error_msg = ''
                else:
                    num_chars = 0
                    num_lines_for_log = 0
                    error_msg = getattr(result, 'error_message', 'Unknown error')
                    status_field = f"error:{error_msg.replace('|',' ')}"

                logger.info(f"url:{url}|depth:{depth}|{status_field}|chars:{num_chars}|lines:{num_lines_for_log}")

                if not result.success:
                    continue

                markdown_content = result.markdown
                num_chars = len(markdown_content)
                num_lines = markdown_content.count('\n') + 1 if markdown_content else 0
                if num_chars < MIN_CONTENT_LENGTH or num_lines < MIN_CONTENT_LINES:
                    logger.info(f"Skipping {url}: content too short (chars: {num_chars}, lines: {num_lines})")
                    logger.info(f"SKIP_SHORT,{url}")
                    continue

                if LANGUAGE and LANGUAGE.strip():
                    try:
                        sample_text = markdown_content[:1000] if len(markdown_content) > 1000 else markdown_content
                        detected_language = detect(sample_text)
                        if detected_language != LANGUAGE.strip():
                            logger.info(f"SKIP_LANGUAGE,{url}")
                        if detected_language != LANGUAGE.strip():
                            logger.info(f"Skipping {url}: Detected language '{detected_language}' != required '{LANGUAGE}'")
                            continue
                    except LangDetectException:
                        logger.info(f"Skipping {url}: Language could not be detected, skipping to be safe.")
                        continue

                save_markdown(markdown_content, CWD, url)
                saved_count += 1
                logger.info(f"Successfully saved: {url}")

        logger.info(f"Deep crawl processing completed. Saved pages: {saved_count}")
        return saved_count

    # Execute the async crawl function
    return asyncio.run(crawl())

# ---- Command-line Interface ----
def main():
    """
    Start the crawl process using configuration values from crawllconfig.py.
    All settings are read directly from config for consistency and simplicity.
    """
    
    if USE_URL_LIST:
        # URL List Mode - crawl specific URLs from file
        url_list_path = os.path.join(os.path.dirname(__file__), URL_LIST_FILE)
        if not os.path.exists(url_list_path):
            logger.error(f"URL list file not found: {url_list_path}")
            return
        
        with open(url_list_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        logger.info(f"URL List Mode: Found {len(urls)} URLs to crawl from {URL_LIST_FILE}")
        total_saved = 0
        
        # Temporarily override CRAWL_URL for each URL and call deep_crawl_urls
        global CRAWL_URL
        original_url = CRAWL_URL
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing URL {i}/{len(urls)}: {url}")
            CRAWL_URL = url
            try:
                saved_pages = deep_crawl_urls()
                total_saved += saved_pages
                logger.info(f"URL {i} completed. Saved: {saved_pages}")
            except Exception as e:
                logger.error(f"Error crawling {url}: {str(e)}")
        
        CRAWL_URL = original_url  # Restore original
        logger.info(f"URL List crawl completed. Total saved pages: {total_saved}")
        
    else:
        # Standard Deep Crawl Mode
        logger.info(f"Starting deep crawl with config: output_dir={CWD}, max_depth={MAX_CRAWL_DEPTH}")
        try:
            saved_pages = deep_crawl_urls()
            logger.info(f"Crawl completed. Saved pages: {saved_pages}")
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"An error occurred during crawling: {str(e)}")

if __name__ == "__main__":
    main() 