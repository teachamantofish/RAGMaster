
Add UI for configuration of each step
Add config files for conf of each step
 

update docs based on context PDF

Add tuning computer tab
Add parameterization of LLM tab







extend crawled doc list
update get PDF script from llamaindex to not use "fast" and verify conttent quality
set up chron job to get docs from llamaparse

Build context composer
- build a multiagent verifier that run just before adding vector data to context
- Add mode selector: 
  - Edit and run code in terminal. Read and fix errors automatically. 
  - Edit and run code but let the user review errors
  - Edit files but don't run anything. 
  - Suggest fixes and don't edit any files. 
  - Keep answers short: 5 lines only 
- Output toml
- add context counter (current + total)

## Resources

- https://www.youtube.com/watch?v=c5jHhMXmXyo&t=173s
- use flowindex: 
  - https://www.youtube.com/watch?v=MU6jA0rUlFY&t=323s
  - https://flowmaker.llamaindex.ai/
- Read Context Engineering.pdf
- Find Context creation diagram. 
- multi agent/agent workflow: https://www.youtube.com/watch?v=MmiveeGxfX0    https://www.youtube.com/watch?v=AxW8gIQ-z5Y&t=98s