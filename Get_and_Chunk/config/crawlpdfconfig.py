# ============================
# PDF PIPELINE CONTROL TOGGLES
# ============================
RUN_DOCLING = True    # Convert PDF to markdown with Docling
RUN_TOC_FIXER = True  # Run script to create markdown headings with toc.json 
RUN_CODE_FIXER = True # Add a lang ID to code fences and pretty-print code blocks

# ============================
# PDF TRIM CONFIGURATION
# ============================
RUN_PDF_TRIM = True             # Remove unwanted pages and save the TOC as toc.json
REMOVE_FRONT     =  0           # remove the title page
TOC_RANGE        = (1, 69)      # Extract these pages into toc.json (1-based inclusive)
REMOVE_RANGES    = [(7, 77)]    # Add more ranges as needed; e.g. [(4, 12), (217, 234)]
REMOVE_BACK      =  1           # Pages to remove from end

