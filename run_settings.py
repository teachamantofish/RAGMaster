from pathlib import Path

# Single place to choose what this run processes.
CWD = Path(r"C:\GIT\Z_Master_Rag\Data\framemaker\run")

# All script logs write here.
LOG_DIR = Path(r"C:\GIT\Z_Master_Rag\Logger\logs")

# Shared run metadata for scripts that require metadata keys.
METADATA = {
    "ID": "",
    "PARSER": "",
    "CRAWL_URL": "",
    "BASE_DIR": "",
    "METADATA_TITLE": "",
    "METADATA_AUTHOR": "",
    "METADATA_CATEGORY": "",
    "METADATA_DESCRIPTION": "",
}
