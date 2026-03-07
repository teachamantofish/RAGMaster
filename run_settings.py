from pathlib import Path

#ID,PARSER,CRAWL_URL,BASE_DIR,METADATA_TITLE,METADATA_AUTHOR,METADATA_CATEGORY,METADATA_DESCRIPTION,TAGS,METADATA_DATE,PAGES
# 0,crawlweb,"https://helpx.adobe.com/adobe-connect/webservices/","connect","Adobe Connect Web Services","Adobe","Adobe Connect","Web services documentation for Adobe Connect","[]","03/07/26",""


# Single place to choose what this run processes.
CWD = Path(r"C:\\\\GIT\\\\Z_Master_Rag\\\\Data")

# All script logs write here.
LOG_DIR = Path(r"C:\\\\GIT\\\\Z_Master_Rag\\\\Logger\\\\logs")

# Shared run metadata for scripts that require metadata keys.
METADATA = {
    "ID": "0",
    "PARSER": "crawlpdf",
    "CRAWL_URL": "C:/GIT/Z_Master_Rag/Data/",
    "BASE_DIR": "connect",
    "METADATA_TITLE": "Adobe Connect Web Services",
    "METADATA_AUTHOR": "Adobe",
    "METADATA_CATEGORY": "Adobe Connect",
    "METADATA_DESCRIPTION": "Adobe Connect Web Services",
    "TAGS": "Adobe Connect, Web Services",
    "METADATA_DATE": "03/07/26",
    "PAGES": "",
}


