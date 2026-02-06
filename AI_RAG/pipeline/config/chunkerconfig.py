import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")  # For OpenAI models: Counts the tokens of all chunk types
MAX_TOKENS_FOR_NODE = 500
CHUNK_SIZE_RANGE = f"max={MAX_TOKENS_FOR_NODE}"
CHUNK_MODEL = "heading-safe"  # Custom identifier: "heading-safe" | "greedy" | "fixed"
CODE_LENGTH = 3  # The number of lines that makes the code chunks "meaningful".
KEYWORD_DENSITY = 0.2  # Keyword density threshold for chunking
ENABLE_CODE_EXTRACTION = True  # Set to False to skip component extraction during testing
OUTPUT_NAME = "a_chunks.json"