# Overview

Code faster, better, and cheaper: 

- Intelligent web scraping: Tuned web scraping of any document type. 
- Chunking tuned for technical documents.
- Vectorize document chunks stored in open source dB and tuned for specific domains. 
- dB of rich sources for tuning context and prompts so my coding agent can behave as a domain expert.
- Agentic rag: example: https://www.youtube.com/watch?v=c5jHhMXmXyo&t=173s and blog: https://weaviate.io/blog/what-is-agentic-rag

## Goals

- Future proof the pipeline: the technology "best" changes every 3 months or so
- Model agnostic: Models evolve and enterprise requirements change: best vs fast, free vs paid, etc.
- Data agnostic: Adaptibility that server any domain or multiple domains
- Strategy and goal agnostic
- Easily tweaked with UI: Don't gate changes with heavy engineering requirements
- Transparent flow and configuration: easy to understand
- Portability
- Free, open source tech stack
- Run locally on cheap hardware
- Modular: It should be easy to run, test, update, conifure, or replace any component without impacting other components

> **Note**
> At a high level these documents are up to date. However, I'm not trying to keep all the details current because the scripts change daily as a result of testing and encountering new issues with newly added docs.

## Architecture

### Decisions

- Convert only markdown. Convert all file types to markdown. 
- Target a 400 token chunk size, but don't be too picky. Source docs are random in quality and nature. *Usually* split code chunks; keep code together unless query responses are imprecise.
- Design a schema that supports future use of re-ranking and graphrag.
- Don't use any code or rely on any Youtube video more than 4 months old!
- Use only free and open source tools: Python, Crawl4AI, LlamaIndex, PostgreSQL
- - Place parameters in separate config files so settings and script execution can be controlled via a web UI.
- Pipeline phases will be in discrete scripts: 
    - crawl_config.py: pipeline configs
    - crawler.py: get and preprocess data. Convert all formats to markdown
    - chunker.py: Chunk, summarize, and create embeddings
    - vector.py: Push to database
    - query.py: Get answers from the database


## RAG pipeline tech stack

**Pipeline**: Crawl > Preprocess (llamaindex + regex) > Chunk (llamaindex) > postprocess (llamaindex) > embedding (OpenAI via (llamaindex) > vector dB (pgvector) > custom web UI

- Crawler: Crawl4AI 
- Chunker: `llamaindex MarkdownNodeParser`
- Vector Embeddings: gpt-3.5-turbo
- Vector Database: pgvector (postgresql)
- Query LLM: TBD
- Front end: TBD: HTML

## Future

- Crawler options: investigate alternatives from [llamaindex docs](https://docs.llamaindex.ai/en/stable/examples/data_connectors/WebPageDemo/): consider crawl4ai for initial crawling and BeautifulSoup for md cleanup. 
- Content consumption support: 
  - DONE: Support PDF 
  - DONE: Support Word
- Investigate Docling, Pandoc, unstructured.io
- Improve dB with pgscale or timescale
- Add llamaindex reranking
- Add agentic rag
- web app UI.
