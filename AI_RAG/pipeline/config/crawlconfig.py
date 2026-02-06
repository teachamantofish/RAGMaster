# ====== Crawling source ====== 
WEB_CRAWL = "Get any web page or website. This method requires lots of configuration."
GIT_CRAWL = "Get a Github doc repo. The dir structure is preserved. Files are already in Markdown format."

# ======  Crawling behavior and limits ====== 
MAX_CRAWL_DEPTH = 2 # Crawl X levels deep counting from the starting URL, not the filesystem hierarchy.
MAX_URLS = 500  # Maximum number of URLs to crawl
MIN_CONTENT_LENGTH = 100  # Number of required characters; else, skip page
MIN_CONTENT_LINES = 2 # Number of required lines (crawl4AI starts the count at 0); else, skip page
INCLUDE_EXTERNAL_DOMAIN = False  # Don't allow crawling external domains
LANGUAGE = "en" # Skip other languages
URL_PATTERN_FILTERS = ["helpx.adobe.com/adobe-connect"] # A CSV list of URL patterns to match (one per line)

# ====== URL list mode ======  
USE_URL_LIST = True  # Use a predefined URL list instead of deep crawling
URL_LIST_FILE = "crawlurls.md"  # A list of URLs to crawl (one per line)

# ====== Post processing options ====== 
CUSTOM_REGEX = True
ADDLINES_TO_CODEBLOCKS = True  # Add the line the preceding the code block as a code commentl.

# ====== Markdown generator options: https://docs.crawl4ai.com/api/parameters/ ======
IGNORE_LINKS = True
IGNORE_IMAGES = True
ESCAPE_HTML = False
BODY_WIDTH = 0
SKIP_INTERNAL_LINKS = True
INCLUDE_SUP_SUB = False
HEADING_STYLE = "ATX"
LIST_STYLE = "dash"
STRIP_COMMENTS = True
PRESERVE_TABLES = True
COLLAPSE_WHITESPACE = False
STRIP_FOOTNOTES = True
STRIP_CODE_BLOCKS = False
STRIP_BLOCKQUOTES = False
STRIP_MATH = False

# ====== CrawlerRunConfig parameters ======
MARKDOWN_GENERATOR = "md_generator"
CACHE_MODE=CacheMode.BYPASS
EXCLUDED_TAGS = ["form", "script", "style", "footer", "header", "rustdoc-toolbar"] # OMIT tags with needed links (e.g. nav)
REMOVE_FORMS = True # Specifically strip <form> elements
REMOVE_OVERLAY_ELEMENTS = True # Attempt to remove modals/popups
CSS_SELECTOR = ".main"  # Get this region only. Wildcards examples: '[id^="ad_"], [class*="sponsor"]'. Use any CSS selector.
EXCLUDED_SELECTOR = ".tracker, .feedback-modal-content, .menu, #menu, .sidebar, #sidebar, .content__toc, #toc_sidebar"  # [class*="modal"]
EXCLUDE_EXTERNAL_LINKS = True # Remove external links from final content
EXCLUDE_SOCIAL_MEDIA_LINKS = True # Remove links to known social sites
EXCLUDE_DOMAINS = ["ads.example.com"] # Exclude links to these domains
EXCLUDE_SOCIAL_MEDIA_DOMAINS = ["facebook.com","twitter.com"] # Extend the default list
