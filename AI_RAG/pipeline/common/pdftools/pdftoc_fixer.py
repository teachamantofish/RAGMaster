# pdftoc_fixer.py
import re
import json
import difflib
import unicodedata
from pathlib import Path

# ---------- helpers ----------
def _strip_leading_numbers(s: str) -> str:
    """
    Remove leading numbering patterns from heading text.
    
    Strips patterns like:
    - "1.2.3 Title" → "Title"
    - "A.1 Title" → "Title" 
    - "IV-2 Title" → "Title"
    - "1. Title" → "Title"
    
    Args:
        s: Input string that may contain leading numbers/letters
        
    Returns:
        String with leading numbering patterns removed
    """
    return re.sub(r'^\s*(?:\d+|[IVXLCDM]+|[A-Z])(?:[\.\-–]\d+)*[\.\-–]?\s+', '', s, flags=re.IGNORECASE)

def _normalize(s: str) -> str:
    """
    Normalize a string for consistent comparison and matching.
    
    This normalization process:
    1. Applies Unicode NFKC normalization
    2. Strips leading numbers/letters
    3. Converts to lowercase
    4. Removes punctuation and replaces with spaces
    5. Converts underscores to spaces
    6. Collapses multiple spaces to single spaces
    
    Args:
        s: Input string to normalize
        
    Returns:
        Normalized string suitable for fuzzy matching
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = _strip_leading_numbers(s)
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = s.replace('_', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _load_toc_json(json_path) -> dict:
    """
    Load TOC data from JSON file and create title-to-level mapping.
    
    Processes TOC entries to create a mapping where:
    - Key: normalized title text
    - Value: heading level (1-6)
    
    For duplicate titles, keeps the shallowest (smallest) level.
    
    Args:
        json_path: Path to the TOC JSON file
        
    Returns:
        Dictionary mapping normalized titles to their heading levels
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    title_to_level = {}
    
    # Process each TOC entry
    for item in data:
        title = item.get("title", "")
        level = int(item.get("level", 2))
        key = _normalize(title)
        if not key:
            continue
        # For duplicate titles, prefer the shallowest level
        title_to_level[key] = min(level, title_to_level.get(key, level))
    return title_to_level

def _best_level_for(title: str, title_to_level: dict, fuzzy: bool, cutoff: float):
    """
    Determine the best heading level for a given title.
    
    First tries exact match on normalized title. If that fails and fuzzy matching
    is enabled, tries to find close matches using difflib.
    
    Args:
        title: The heading title to look up
        title_to_level: Dictionary mapping normalized titles to levels
        fuzzy: Whether to enable fuzzy matching for close matches
        cutoff: Minimum similarity ratio for fuzzy matches (0.0-1.0)
        
    Returns:
        Heading level (int) if match found, None otherwise
    """
    key = _normalize(title)
    if not key:
        return None
    if key in title_to_level:
        return title_to_level[key]
    if fuzzy:
        candidates = list(title_to_level.keys())
        close = difflib.get_close_matches(key, candidates, n=1, cutoff=cutoff)
        if close:
            return title_to_level[close[0]]
    return None

def _build_level1_insertion_map(toc_data: list) -> dict:
    """
    Build a mapping of where level 1 headings should be inserted.
    
    Analyzes the TOC structure to determine where missing level 1 headings
    should be inserted. Creates mappings for both normalized titles and
    original titles with numbers to handle cases like:
    - "2.1 Overview" and "9.1 Overview" (both normalize to "overview")
    
    Args:
        toc_data: List of TOC entries from JSON (each with level, title, page)
        
    Returns:
        Dictionary where keys are normalized/original titles of level 2+ headings
        and values are the level 1 heading titles that should precede them
    """
    insertion_map = {}
    current_level1 = None
    
    # Walk through TOC entries to find level 1 → level 2+ transitions
    for item in toc_data:
        level = int(item.get("level", 2))
        title = item.get("title", "")
        
        if level == 1:
            # Found a level 1 heading - remember it
            current_level1 = title
        elif level >= 2 and current_level1:
            # This is the first level 2+ heading after a level 1
            # The level 1 should be inserted before this heading
            key = _normalize(title)
            original_key = title.lower().strip()
            
            # Add mappings for both normalized and original versions
            # This handles cases where multiple headings normalize to the same thing
            if key and key not in insertion_map:
                insertion_map[key] = current_level1
            if original_key and original_key not in insertion_map:
                insertion_map[original_key] = current_level1
                
            current_level1 = None  # Only insert once per level 1 section
    
    return insertion_map

# ...existing code...

def rebuild_md_headings_from_toc(
    toc_json_path,
    md_in_path,
    md_out_path=None,
    fuzzy: bool = True,
    cutoff: float = 0.90,
    max_level: int = 6,
    preserve_existing: bool = False,
):
    """
    Rebuild markdown headings from TOC data in linear order.
    
    Processes TOC entries sequentially and only converts lines that match 
    the expected next TOC entry. This prevents duplicates and ensures 
    headings appear in the correct order.
    
    Args:
        toc_json_path: Path to TOC JSON file with heading structure
        md_in_path: Path to input markdown file to process
        md_out_path: Path for output file (defaults to input with .rebuilt.md suffix)
        fuzzy: Enable fuzzy matching for heading titles (default: True)
        cutoff: Minimum similarity for fuzzy matches, 0.0-1.0 (default: 0.90)
        max_level: Maximum heading level to output (default: 6)
        preserve_existing: If True, don't strip existing heading markers (default: False)
        
    Returns:
        String path to the output markdown file
        
    Raises:
        RuntimeError: If TOC JSON is empty or cannot be processed
    """
    # Convert paths to Path objects for consistent handling
    toc_json_path = Path(toc_json_path)
    md_in_path = Path(md_in_path)

    # Load the raw TOC data 
    toc_data = json.loads(toc_json_path.read_text(encoding="utf-8"))
    
    if not toc_data:
        raise RuntimeError("TOC JSON is empty—cannot rebuild headings.")
    
    # Create a queue of TOC entries to process in order
    toc_queue = list(toc_data)
    current_toc_index = 0

    # Read the input markdown file
    text = md_in_path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    out_lines = []

    # State tracking for front matter processing
    in_front_matter = False
    # Regex to match existing heading lines with optional indentation
    heading_re = re.compile(r'^(?P<indent>\s{0,3})(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$')

    # Process each line of the input file
    for idx, line in enumerate(lines):
        # Handle YAML front matter passthrough (preserve as-is)
        if idx == 0 and line.strip() == '---':
            in_front_matter = True
            out_lines.append(line)
            continue
        if in_front_matter:
            out_lines.append(line)
            if line.strip() == '---':
                in_front_matter = False
            continue

        # Start with the original line
        working = line
        
        # If this is an existing heading and we're not preserving, extract just the title
        m = heading_re.match(line)
        if m and not preserve_existing:
            working = m.group('title')

        # Get the candidate text (stripped of whitespace)
        candidate = working.strip()
        
        # Handle empty lines - just pass through
        if not candidate:
            out_lines.append(working)
            continue
            
        # Skip lines that are clearly not headings (code blocks, tables, lists)
        if candidate.startswith('```') or candidate.startswith('|') or re.match(r'^[-*+]\s+\S', candidate):
            out_lines.append(working)
            continue
            
        # Skip very long lines (likely paragraphs, not headings)
        if len(candidate) > 180:
            out_lines.append(working)
            continue

        # Check if we have more TOC entries to process
        if current_toc_index >= len(toc_queue):
            # No more TOC entries, just pass through remaining lines
            out_lines.append(working)
            continue

        # Get the current expected TOC entry
        current_toc_entry = toc_queue[current_toc_index]
        expected_title = current_toc_entry.get("title", "")
        expected_level = int(current_toc_entry.get("level", 2))
        
        # Check if this line matches the expected TOC entry
        matches_expected = False
        
        # First try exact match on normalized text
        candidate_normalized = _normalize(candidate)
        expected_normalized = _normalize(expected_title)
        
        if candidate_normalized and expected_normalized and candidate_normalized == expected_normalized:
            matches_expected = True
        elif fuzzy and candidate_normalized and expected_normalized:
            # Try fuzzy matching
            similarity = difflib.SequenceMatcher(None, candidate_normalized, expected_normalized).ratio()
            if similarity >= cutoff:
                matches_expected = True
        
        if matches_expected:
            # This line matches the expected TOC entry - convert it to a heading
            level = max(1, min(int(expected_level), max_level))
            clean_title = _strip_leading_numbers(expected_title)  # Use the TOC title, not the candidate
            out_lines.append(f"{'#' * level} {clean_title}")
            
            # Move to the next TOC entry
            current_toc_index += 1
        else:
            # This line doesn't match the expected TOC entry - pass it through as-is
            out_lines.append(working)

    # Write the processed content back to the input file (in-place modification)
    md_in_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    return str(md_in_path)

# ...existing code...

"""
DO NOT DELETE THIS
#     Alternative approach: Adjust existing heading levels without rebuilding.
#     
#     This function modifies existing markdown heading lines (# ...) to match 
#     levels from toc.json, but does NOT:
#     - Insert missing level 1 headings
#     - Remove non-heading text that was incorrectly marked as headings
#     - Strip numbering from headings
#     
#     Only modifies lines starting with up to 3 spaces + 1..6 hashes.
#     Kept as reference implementation for cases where full rebuild is not desired.
#     
#     Args:
#         toc_json_path: Path to TOC JSON file
#         md_in_path: Path to input markdown file
#         md_out_path: Path for output (defaults to input with .fixed.md suffix)
#         fuzzy: Enable fuzzy matching for titles
#         cutoff: Similarity threshold for fuzzy matching
#         max_level: Maximum heading level to output
"""