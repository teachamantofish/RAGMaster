from pathlib import Path

# ID,PARSER,CRAWL_URL,BASE_DIR,METADATA_TITLE,METADATA_AUTHOR,METADATA_CATEGORY,METADATA_DESCRIPTION,TAGS,METADATA_DATE,PAGES
# 0,crawlpdf,"C:/GIT/Z_Master_Rag/Data/","mifref","FrameMaker MIF Reference","Adobe","MIF","FrameMaker Maker Interchange Format reference guide","[]","5/1/2023",5-307

# Single place to choose what this run processes.
CWD = Path(r"C:\\\\GIT\\\\Z_Master_Rag\\\\Data")

# All script logs write here.
LOG_DIR = Path(r"C:\\\\GIT\\\\Z_Master_Rag\\\\Logger\\\\logs")

# Shared run metadata for scripts that require metadata keys.
METADATA = {
    "ID": "0",
    "PARSER": "crawlpdf",
    "CRAWL_URL": "C:/GIT/Z_Master_Rag/Data/",
    "BASE_DIR": "mifref",
    "METADATA_TITLE": "FrameMaker MIF Reference",
    "METADATA_AUTHOR": "Adobe",
    "METADATA_CATEGORY": "MIF",
    "METADATA_DESCRIPTION": "FrameMaker Maker Interchange Format reference guide",
    "TAGS": "[tag1, tag2]",
    "METADATA_DATE": "5/1/2023",
    "PAGES": "5-307",
}
