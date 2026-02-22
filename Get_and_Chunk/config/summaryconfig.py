import os
from sqlalchemy import null

# on each run, do the following:
# 1. Check if LLM access is configured; e.g.: is the Ollama server running?
# 2. Verify small chunk handling settings
# 3. Verify global summary settings
# 4. Test on 1-N chunks first

# CHUNK_SUMMARY_MODEL = "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"  # Format: backend:model_name (e.g. prepend local models with "local:")
# CHUNK_SUMMARY_MODEL = "local:qwen2.5:14b-instruct-q8_0"  # Format: backend:model_name (e.g. prepend local models with "local:")


CHUNK_SUMMARY_MODEL = os.getenv("CHUNK_SUMMARY_MODEL", "qwen3-next-80b-a3b-instruct-hmt")
TESTINGMODE = "Null"  # If null, run the full test; if a number, limit to N chunks for faster testing.

# Chunk-level summary
ENABLE_CHUNK_SUMMARY = True  # Set to False to skip summary generation during testing
CHUNK_SUMMARY_SIZE = 100 # Max tokens for chunk summary
CHUNK_SUMMARY_PROMPT_TEMPLATE = "Read the parent heading '{heading_context}' and use that context to summarize the current chunk in {size} tokens or less. Consider how this chunk relates to its parent and provide conceptual insights rather than reworded terminology. If the content is short, provide a higher-level abstraction. Focus on functionality and purpose. Active voice. Be concise and clear."
CHUNK_SUMMARY_TEMPERATURE = 1 # Temperature for OpenAI API calls. Other parameters can be added as needed. Lower values make the output more deterministic.

# Small chunk handling
SKIP_CHUNK_THRESHOLD = 50 # Do not summarize.
PAD_CHUNK_THRESHOLD = 150  # Specifies whether to add the concat_header_path and summaries to the chunk content.
ADD_CONCAT_HEADER_PATH = True  # Whether to add the concat_header_path to the chunk content for small chunks
ADD_CHUNK_SUMMARY = True  # Whether to add the chunk summary to the chunk content for small chunks
ADD_PAGE_SUMMARY = False  # Whether to add the page summary to the chunk content for small chunks

# File/ or H1 summary
ENABLE_FILE_SUMMARY = False  # Disable file and page-level (Heading 1) summaries to avoid adding content that competes with granular chunk content.
FILE_SUMMARY_MODEL = "qwen3-next-80b-a3b-instruct-hmt"
FILE_SUMMARY_SIZE = 125 # Max tokens for page summary
FILE_SUMMARY_PROMPT = (f"Summarize the main ideas in this page or heading in {FILE_SUMMARY_SIZE} tokens. Do not include heading text, bullets, quotes. Identify the functionality and purpose of child nodes. tive voice. Be concise and clear.")
FILE_SUMMARY_TEMPERATURE = 1 # Lower values make the output more deterministic.

# Summarize Summaries
ENABLE_SUMMARY_SUMMARY = False  # Set to False to skip summary generation during testing
CHILD_COUNT = 2 # Threshold for which we create a summary of sibling summaries. 
SUMMARY_SUMMARY_MODEL = "qwen3-next-80b-a3b-instruct-hmt"
SUMMARY_SUMMARY_SIZE = 125 # Max tokens for summary of summaries
SUMMARY_SUMMARY_PROMPT = (f"Summarize the main ideas in this summary of summaries in {SUMMARY_SUMMARY_SIZE} tokens. Do not include heading text, bullets, quotes. Identify the functionality and purpose of this related content. Active voice. Be concise and clear.")
SUMMARY_SUMMARY_TEMPERATURE = 1 # Temperature for OpenAI API calls. Other parameters can be added as needed. Lower values make the output more deterministic.

# Code-level summary
ENABLE_CODE_SUMMARY = False  # Unused. Set to False to skip code summary generation during testing
CODE_SUMMARY_MODEL = "qwen3-next-80b-a3b-instruct-hmt"
CODE_SUMMARY_SIZE = 40 # Max tokens for code summary
CODE_SUMMARY_PROMPT = (f"Summarize this code chunk's purpose in {CODE_SUMMARY_SIZE} tokens: focus on functionality. Active voice. Be concise and clear."
)
CODE_SUMMARY_TEMPERATURE = 0.2 # Temperature for OpenAI API calls. Other parameters can be added as needed. Lower values make the output more deterministic.

# --- Backend/model parser ---
def parse_model_string(model_string):
    """
    Parse model string in format 'backend:model_name'.
    Returns (backend, model_name).
    If no backend specified, defaults to 'openai'.
    """
    if ':' in model_string:
        backend, model_name = model_string.split(':', 1)
    else:
        backend, model_name = 'openai', model_string
    return backend, model_name

# Central lookup so callers can reference summary parameters without repeating imports.
SUMMARY_SETTINGS = {
    "chunk": {
        "backend": parse_model_string(CHUNK_SUMMARY_MODEL)[0],
        "model": parse_model_string(CHUNK_SUMMARY_MODEL)[1],
        "system_prompt_template": CHUNK_SUMMARY_PROMPT_TEMPLATE,  # Template with {heading_context} and {size} placeholders
        "size": CHUNK_SUMMARY_SIZE,
        "temperature": CHUNK_SUMMARY_TEMPERATURE,
    },
    "file": {
        "backend": parse_model_string(FILE_SUMMARY_MODEL)[0],
        "model": parse_model_string(FILE_SUMMARY_MODEL)[1],
        "system_prompt": FILE_SUMMARY_PROMPT,
        "size": FILE_SUMMARY_SIZE,
        "temperature": FILE_SUMMARY_TEMPERATURE,
    },
    "sibling": {
        "backend": parse_model_string(SUMMARY_SUMMARY_MODEL)[0],
        "model": parse_model_string(SUMMARY_SUMMARY_MODEL)[1],
        "system_prompt": SUMMARY_SUMMARY_PROMPT,
        "size": SUMMARY_SUMMARY_SIZE,
        "temperature": SUMMARY_SUMMARY_TEMPERATURE,
    },
    "code": {
        "backend": parse_model_string(CODE_SUMMARY_MODEL)[0],
        "model": parse_model_string(CODE_SUMMARY_MODEL)[1],
        "system_prompt": CODE_SUMMARY_PROMPT,
        "size": CODE_SUMMARY_SIZE,
        "temperature": CODE_SUMMARY_TEMPERATURE,
    },
}

# --- Model-agnostic summary backend dispatch ---
def openai_summary_backend(text, params, **kwargs):
    import openai
    inputs = []
    if params.get("system_prompt"):
        inputs.append({"role": "system", "content": params["system_prompt"]})
    inputs.append({"role": "user", "content": text})
    response = openai.responses.create(
        model=params["model"],
        input=inputs,
        temperature=params.get("temperature", 1),
    )
    # Extract text from OpenAI response (simple version)
    texts = []
    for block in getattr(response, "output", []) or []:
        for item in getattr(block, "content", []) or []:
            text = None
            if hasattr(item, "text"):
                text = item.text
            elif isinstance(item, dict):
                text = item.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()

# Example stub for a local model backend (Unsloth, Qwen3, etc.)
# Ollama only recognizes models it has pulled or built itself.
# You cannot point Ollama to an arbitrary folder of models.
# Use ollama pull or build a custom model with a Modelfile.
def local_summary_backend(text, params, **kwargs):
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": params["model"],
        "prompt": text,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"[OLLAMA ERROR] {e}"

def huggingface_summary_backend(text, params, **kwargs):
    import requests
    endpoint_url = os.getenv("HF_ENDPOINT_URL")
    hf_token = os.getenv("HF_TOKEN")
    
    if not endpoint_url or not hf_token:
        raise ValueError("HF_ENDPOINT_URL and HF_TOKEN must be set for HuggingFace backend")
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    # Build messages for chat completion format (OpenAI-compatible)
    messages = []
    if params.get("system_prompt"):
        messages.append({"role": "system", "content": params["system_prompt"]})
    messages.append({"role": "user", "content": text})
    
    payload = {
        "messages": messages,
        "max_tokens": params.get("size", 100),
        "temperature": params.get("temperature", 1.0),
    }
    
    try:
        # Try OpenAI-compatible chat completions endpoint first
        response = requests.post(f"{endpoint_url}/v1/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            # Get detailed error message
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            return f"[HUGGINGFACE ERROR] {response.status_code}: {error_detail}"
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[HUGGINGFACE ERROR] {e}"

# Dispatch table for summary backends
SUMMARY_BACKENDS = {
    "openai": openai_summary_backend,
    "local": local_summary_backend,  # e.g., Unsloth, Qwen3, etc.
    "huggingface": huggingface_summary_backend,  # HuggingFace inference endpoints
}

# Helper to select and call the backend
def run_summary_backend(backend_name, text, params, **kwargs):
    backend = SUMMARY_BACKENDS.get(backend_name)
    if not backend:
        raise ValueError(f"Unknown summary backend: {backend_name}")
    return backend(text, params, **kwargs)

