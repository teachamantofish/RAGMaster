"""Generate curriculum-ready triplets for the Train Embed Model pipeline.

This script is invoked by `0pipeline_manager.py` after chunking to build all
training/eval splits used downstream (tokenizer, fine-tuner, diagnostics). It
does the following:

1. Load curated chunks (plus metadata) emitted by the upstream crawl/clean
    stages.
2. Engineer multiple positive-pair views (parent-child, sibling, summary, and
    section matches) while honoring project-specific exclusions.
3. For every positive pair, sample easy/medium/hard negatives with steadily
    tighter cosine thresholds so we can teach progressive difficulty during
    curriculum training.
4. Enforce the configured category mix, same-source ratio, and per-difficulty
    quotas before exporting JSON train/test splits (and stats) into
    `TRAINING_DATA_DIR/<difficulty>/`.

Maintaining this docstring alongside the pipeline docs keeps feature work in
sync with the automation that publishes triplets to the rest of the stack."""

# Filtering overview (read top-to-bottom while following `main()`):
# - Global chunk gates: `filter_chunks_for_training()` removes low-quality
#   content and marks chunks with summaries; `_is_excluded_anchor_chunk()` blocks
#   disallowed types (tables by default) from anchor/positive roles.
# - Structural safety rails: `_generate_parent_child_pairs()` enforces the
#   parent/child rule, while sibling/section generators keep anchors aligned
#   with meaningful context before balancing/ratio enforcement steps.
# - Table throttling: `_initialize_table_chunk_cache()` +
#   `_table_chunk_allowed_for_medium()` limit how many table rows survive as
#   medium negatives under a single heading so we don't flood the dataset.
# - Difficulty-specific filters: within `create_training_triplets_by_difficulty()`
#   every easy/medium/hard negative is scored via `_negative_similarity_...` and
#   rejected when cosine similarity breaks that tier's bounds; the per-tier
#   logic also encodes rules like "easy must change H1" or "hard must share
#   scope" before the triplet gets written.


import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations
from sentence_transformers import SentenceTransformer
from scripts.config_training_embed import *
from scripts.custom_logger import setup_global_logger


# Temporary filter to keep specific chunk types from acting as anchors/positives.
chunkfile = BASE_CWD / "a_chunks.json"
EXCLUDED_ANCHOR_POSITIVE_TYPES = {
    t.lower() for t in TRAINING_CONFIG.get("excluded_anchor_positive_types", ["table"])
    if isinstance(t, str)
}

# Set up global logger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Triplets Generated", "Training Data Type"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)

MAX_TABLE_MEDIUM_NEGATIVES_PER_HEADING = 3
HARD_CANDIDATE_SAMPLE_LIMIT = 20
HARD_PATH_SIM_THRESHOLD = 0.35
HARD_CONTENT_SIM_THRESHOLD = 0.20
HARD_MIN_PATH_FLOOR = 0.10
HARD_MIN_CONTENT_FLOOR = 0.05
_TABLE_CHUNK_CACHE_READY = False
_TABLE_CHUNK_IDS_BY_HEADING: Dict[str, List[str]] = {}
_ALLOWED_TABLE_CHUNK_IDS_BY_HEADING: Dict[str, Set[str]] = {}


def _chunk_category(chunk: Dict) -> str:
    """Return a normalized category label for a chunk."""
    return (chunk.get("category") or "unknown").lower()


def _top_level_heading(chunk: Dict) -> str:
    """Return the normalized top-level (H1) heading for the chunk."""
    header_path = (chunk.get("concat_header_path") or "").strip()
    if header_path:
        return header_path.split("/")[0].strip().lower()
    fallback = (chunk.get("heading") or chunk.get("title") or "").strip()
    return fallback.lower()


def _is_different_leaf(anchor_leaf: str, candidate: Dict) -> bool:
    """Return True when the candidate's leaf heading differs from the anchor's."""
    candidate_leaf = (candidate.get("heading") or candidate.get("concat_header_path") or "").lower()
    return bool(anchor_leaf and candidate_leaf and candidate_leaf != anchor_leaf)


def _is_excluded_anchor_chunk(chunk: Dict) -> bool:
    """True when chunk_type is blocked from anchor/positive roles."""
    chunk_type = chunk.get("chunk_type")
    return bool(chunk_type and chunk_type.lower() in EXCLUDED_ANCHOR_POSITIVE_TYPES)


def _is_table_chunk(chunk: Dict) -> bool:
    """Return True when the chunk is a table (used for medium-neg filters)."""
    chunk_type = chunk.get("chunk_type")
    return isinstance(chunk_type, str) and chunk_type.lower() == "table"


def _initialize_table_chunk_cache(all_chunks: List[Dict]) -> None:
    """Cache table chunk ids per heading so we can throttle medium-neg sampling."""
    global _TABLE_CHUNK_CACHE_READY
    if _TABLE_CHUNK_CACHE_READY:
        return
    for chunk in all_chunks:
        if not _is_table_chunk(chunk):
            continue
        heading = _top_level_heading(chunk)
        chunk_id = chunk.get("id")
        if not heading or not chunk_id:
            continue
        _TABLE_CHUNK_IDS_BY_HEADING.setdefault(heading, []).append(chunk_id)
    limit = MAX_TABLE_MEDIUM_NEGATIVES_PER_HEADING
    for heading, ids in _TABLE_CHUNK_IDS_BY_HEADING.items():
        if len(ids) > limit:
            _ALLOWED_TABLE_CHUNK_IDS_BY_HEADING[heading] = set(random.sample(ids, limit))
        else:
            _ALLOWED_TABLE_CHUNK_IDS_BY_HEADING[heading] = set(ids)
    _TABLE_CHUNK_CACHE_READY = True


def _table_chunk_allowed_for_medium(chunk: Dict, heading: str) -> bool:
    """Check whether a table chunk under the heading is within the sampling cap."""
    if not _is_table_chunk(chunk):
        return True
    chunk_id = chunk.get("id")
    if not chunk_id:
        return True
    allowed_ids = _ALLOWED_TABLE_CHUNK_IDS_BY_HEADING.get(heading)
    if allowed_ids is None:
        return True
    return chunk_id in allowed_ids


def _split_header_segments(path_value: str) -> List[str]:
    """Split a concat header path into normalized segments for comparisons."""
    if not path_value:
        return []
    return [segment.strip().lower() for segment in path_value.split("/") if segment.strip()]


def _shares_heading_scope(anchor_segments: List[str], candidate_path: str, candidate_heading: str) -> bool:
    """Return True when candidate shares the required heading scope with anchor."""
    if not anchor_segments:
        return True
    candidate_segments = _split_header_segments(candidate_path)
    if not candidate_segments and candidate_heading:
        candidate_segments = [candidate_heading.strip().lower()]
    if not candidate_segments:
        return False
    if len(anchor_segments) >= 2:
        if len(candidate_segments) < 2:
            return False
        return candidate_segments[:2] == anchor_segments[:2]
    return candidate_segments[0] == anchor_segments[0]


def _length_bucket(content_len: int) -> str:
    """Map raw content length to a coarse bucket label."""
    if content_len < 200:
        return "short"
    if content_len < 800:
        return "medium"
    return "long"


def _normalize_ratio_map(ratio_map: Dict[str, float]) -> Dict[str, float]:
    """Normalize a ratio dictionary so values sum to 1.0."""
    positive_items = {k.lower(): v for k, v in ratio_map.items() if v > 0}
    total = sum(positive_items.values())
    if not positive_items or total <= 0:
        return {}
    return {k: v / total for k, v in positive_items.items()}


COSINE_THRESHOLD_MAP = {
    "easy": EASY_NEGATIVE_MAX_COSINE,
    "medium": MEDIUM_NEGATIVE_MAX_COSINE,
    "hard": HARD_NEGATIVE_MAX_COSINE,
}

COSINE_MIN_THRESHOLD_MAP = {
    "easy": None,
    "medium": MEDIUM_NEGATIVE_MIN_COSINE,
    "hard": None,
}

# Get the batch size and fallback to a safe default if not set or invalid.
CHUNK_EMBED_BATCH_SIZE = TRAINING_CONFIG.get("chunk_embedding_batch_size", 32)
_EMBED_MODEL: Optional[SentenceTransformer] = None


def _get_embedding_vector(chunk: Dict) -> Optional[List[float]]:
    """Return a normalized embedding vector for the chunk, if available."""
    candidate_keys = ("embedding", "vector", "embedding_vector")
    for key in candidate_keys:
        raw_vector = chunk.get(key)
        if raw_vector in (None, ""):
            continue
        vector = raw_vector
        if isinstance(vector, str):
            stripped = vector.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
                vector = parsed
            except json.JSONDecodeError:
                try:
                    vector = [float(piece) for piece in stripped.replace(",", " ").split()]
                except ValueError:
                    continue
        if isinstance(vector, (list, tuple)):
            cleaned: List[float] = []
            valid = True
            for value in vector:
                if isinstance(value, (int, float)):
                    cleaned.append(float(value))
                elif isinstance(value, str):
                    try:
                        cleaned.append(float(value.strip()))
                    except ValueError:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid and cleaned:
                return cleaned
    return None


def _chunk_embedding_text(chunk: Dict) -> str:
    """Return the text representation used when computing chunk embeddings."""
    content = (chunk.get("content") or "").strip()
    summary = (chunk.get("chunk_summary") or "").strip()
    heading = (chunk.get("concat_header_path") or "").strip()
    if summary and summary.lower() != "false" and len(summary) > 20:
        return f"{heading}: {summary}" if heading else summary
    if heading and len(content) > 100:
        return f"{heading}: {content}"
    return content


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model used for chunk vectors."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        model_source = CONFIG_MODEL_NAME if Path(CONFIG_MODEL_NAME).exists() else BASE_MODEL
        logger.info("Loading embedding model from %s for chunk vector generation", model_source)
        _EMBED_MODEL = SentenceTransformer(str(model_source))
    return _EMBED_MODEL


def _ensure_chunk_embeddings(chunks: List[Dict]) -> None:
    """Compute embeddings for chunks missing vectors so cosine checks can run."""
    missing_chunks: List[Dict] = []
    texts: List[str] = []
    for chunk in chunks:
        if _get_embedding_vector(chunk):
            continue
        text = _chunk_embedding_text(chunk)
        if not text:
            continue
        missing_chunks.append(chunk)
        texts.append(text)
    if not missing_chunks:
        return
    model = _get_embedding_model()
    logger.info("Computing embeddings for %d chunks missing vectors", len(missing_chunks))
    for start in range(0, len(missing_chunks), CHUNK_EMBED_BATCH_SIZE):
        end = start + CHUNK_EMBED_BATCH_SIZE
        batch_texts = texts[start:end]
        batch_chunks = missing_chunks[start:end]
        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=CHUNK_EMBED_BATCH_SIZE,
        )
        for chunk, vector in zip(batch_chunks, embeddings):
            if hasattr(vector, "tolist"):
                cleaned = [float(v) for v in vector.tolist()]
            else:
                cleaned = [float(v) for v in vector]
            chunk["embedding"] = cleaned


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} != {len(vec_b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def _negative_similarity_exceeds_threshold(
    anchor: Dict, negative: Dict, difficulty: str
) -> Tuple[bool, Optional[float]]:
    """Return True when anchor/negative cosine similarity violates the difficulty range."""
    difficulty_key = (difficulty or "").lower()
    if difficulty_key not in COSINE_THRESHOLD_MAP:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    anchor_vec = _get_embedding_vector(anchor)
    negative_vec = _get_embedding_vector(negative)
    if not anchor_vec or not negative_vec:
        return False, None
    if len(anchor_vec) != len(negative_vec):
        logger.debug(
            "Skipping cosine filter due to length mismatch for anchor %s", anchor.get("id")
        )
        return False, None
    similarity = _cosine_similarity(anchor_vec, negative_vec)
    min_threshold = COSINE_MIN_THRESHOLD_MAP.get(difficulty_key)
    if min_threshold is not None and similarity < min_threshold:
        return True, similarity
    return similarity > COSINE_THRESHOLD_MAP[difficulty_key], similarity


def load_chunks() -> Tuple[List[Dict], Dict]:
    """Load chunks from JSON file and return chunks list plus metadata."""
    with open(chunkfile, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    
    if isinstance(loaded, dict) and "chunks" in loaded:
        chunks = loaded["chunks"]
        metadata = {k: v for k, v in loaded.items() if k != "chunks"}
    else:
        chunks = loaded
        metadata = {}
    
    logger.info(f"Loaded {len(chunks)} chunks from {chunkfile}")
    return chunks, metadata


def filter_chunks_for_training(chunks: List[Dict]) -> List[Dict]:
    """Filter chunks suitable for training based on content length and quality."""
    # GLOBAL FILTERING PASS: This is the first gate that drops unusable chunks
    # (length, empty content, placeholder summaries) before any positive/negative
    # logic runs. Everything downstream assumes only these survivors remain.
    filtered = []
    
    for chunk in chunks:
        content = chunk.get("content", "").strip()
        
        # Skip chunks that are too short or too long
        if len(content) < MIN_CONTENT_LENGTH or len(content) > MAX_CONTENT_LENGTH:
            continue
            
        # Skip chunks without meaningful content
        if not content or content.lower() in ["", "n/a", "none", "null"]:
            continue
            
        # Prefer chunks with summaries as they tend to be higher quality
        chunk_summary = chunk.get("chunk_summary", "")
        if chunk_summary and chunk_summary.lower() != "false":
            chunk["has_summary"] = True
        else:
            chunk["has_summary"] = False
            
        filtered.append(chunk)
    
    logger.info(f"Filtered to {len(filtered)} chunks suitable for training")
    return filtered


def group_chunks_by_domain(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """Group chunks by source category (and its length buckets) only."""
    groups = defaultdict(list)
    
    for chunk in chunks:
        category = _chunk_category(chunk)
        content_len = len(chunk.get("content", ""))
        length_bucket = _length_bucket(content_len)
        base_key = f"category::{category}"
        groups[base_key].append(chunk)
        groups[f"{base_key}::{length_bucket}"].append(chunk)
    
    filtered_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    preview = list(filtered_groups.keys())[:10]
    logger.info(
        "Grouped chunks into %d domains (category + length only): %s%s",
        len(filtered_groups),
        preview,
        "..." if len(filtered_groups) > len(preview) else "",
    )
    return filtered_groups


def generate_positive_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict, str]]:
    """Generate positive pairs from chunks using various similarity strategies."""
    positive_pairs = []
    
    # Strategy 1: Parent-child relationships
    # Enforces the "parent rule" so curriculum training always sees canonical
    # hierarchical pairs before we add looser similarity heuristics.
    parent_child_pairs = _generate_parent_child_pairs(chunks)
    positive_pairs.extend([(p, c, "parent_child") for p, c in parent_child_pairs])
    
    # Strategy 2: Same heading path (siblings)
    sibling_pairs = _generate_sibling_pairs(chunks)
    positive_pairs.extend([(s1, s2, "siblings") for s1, s2 in sibling_pairs])
    
    # Strategy 3: Similar summaries
    summary_pairs = _generate_summary_similarity_pairs(chunks)
    positive_pairs.extend([(s1, s2, "similar_summary") for s1, s2 in summary_pairs])
    
    # Strategy 4: Same document sections
    section_pairs = _generate_same_section_pairs(chunks)
    positive_pairs.extend([(s1, s2, "same_section") for s1, s2 in section_pairs])
    
    if EXCLUDED_ANCHOR_POSITIVE_TYPES:
        before_filter = len(positive_pairs)
        positive_pairs = [
            pair
            for pair in positive_pairs
            if not _is_excluded_anchor_chunk(pair[0]) and not _is_excluded_anchor_chunk(pair[1])
        ]
        filtered_out = before_filter - len(positive_pairs)
        if filtered_out:
            logger.info(
                "Excluded %d positive pairs where chunk_type matched %s",
                filtered_out,
                ", ".join(sorted(EXCLUDED_ANCHOR_POSITIVE_TYPES)),
            )

    logger.info(
        f"Generated {len(positive_pairs)} positive pairs",
        extra={"Triplets Generated": str(len(positive_pairs)), "Training Data Type": "positive_pairs"},
    )
    balanced_pairs = _balance_anchor_categories(positive_pairs)
    balanced_pairs = _rebalance_same_source_pairs(balanced_pairs)
    return balanced_pairs


def _generate_parent_child_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from parent-child relationships."""
    id_to_chunk = {chunk["id"]: chunk for chunk in chunks if chunk.get("id")}
    pairs = []
    
    for chunk in chunks:
        parent_id = chunk.get("parent_id")
        if parent_id and parent_id in id_to_chunk:
            parent = id_to_chunk[parent_id]
            pairs.append((parent, chunk))
    
    return pairs


def _generate_sibling_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from chunks with same header path (siblings)."""
    by_header_path = defaultdict(list)
    
    for chunk in chunks:
        header_path = chunk.get("concat_header_path", "")
        if header_path and len(header_path) > 10:  # Avoid very short/generic paths
            by_header_path[header_path].append(chunk)
    
    pairs = []
    for path, path_chunks in by_header_path.items():
        if len(path_chunks) >= 2:
            # Generate pairs within the same header path
            for chunk1, chunk2 in combinations(path_chunks[:5], 2):  # Limit to avoid explosion
                pairs.append((chunk1, chunk2))
    
    return pairs


def _generate_summary_similarity_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from chunks with similar summaries."""
    summary_groups = defaultdict(list)
    
    for chunk in chunks:
        summary = chunk.get("chunk_summary", "").strip()
        if summary and summary.lower() != "false" and len(summary) > 20:
            # Group by first few words of summary to find similar concepts
            summary_key = " ".join(summary.lower().split()[:5])
            summary_groups[summary_key].append(chunk)
    
    pairs = []
    for key, group_chunks in summary_groups.items():
        if len(group_chunks) >= 2:
            for chunk1, chunk2 in combinations(group_chunks[:3], 2):  # Limit combinations
                pairs.append((chunk1, chunk2))
    
    return pairs


def _generate_same_section_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from chunks in the same document section."""
    by_section = defaultdict(list)
    
    for chunk in chunks:
        filename = chunk.get("filename", "")
        # Use first part of header path as section identifier
        header_path = chunk.get("concat_header_path", "")
        if header_path and "/" in header_path:
            section = header_path.split("/")[0]
            section_key = f"{filename}:{section}"
            by_section[section_key].append(chunk)
    
    pairs = []
    for section_key, section_chunks in by_section.items():
        if len(section_chunks) >= 2:
            for chunk1, chunk2 in combinations(section_chunks[:4], 2):
                pairs.append((chunk1, chunk2))
    
    return pairs


def _balance_anchor_categories(
    positive_pairs: List[Tuple[Dict, Dict, str]]
) -> List[Tuple[Dict, Dict, str]]:
    """Down-sample anchors to approximate the configured category mix."""
    if not positive_pairs or not ANCHOR_CATEGORY_BALANCE:
        return positive_pairs
    normalized_targets = _normalize_ratio_map(ANCHOR_CATEGORY_BALANCE)
    if not normalized_targets:
        return positive_pairs
    buckets: Dict[str, List[Tuple[Dict, Dict, str]]] = defaultdict(list)
    for pair in positive_pairs:
        buckets[_chunk_category(pair[0])].append(pair)
    missing = [cat for cat in normalized_targets if not buckets.get(cat)]
    if missing:
        logger.warning(
            "Cannot balance anchors; missing categories: %s",
            ", ".join(sorted(missing)),
        )
        return positive_pairs
    for bucket in buckets.values():
        random.shuffle(bucket)
    try:
        max_total = min(len(buckets[cat]) / ratio for cat, ratio in normalized_targets.items())
    except ZeroDivisionError:
        return positive_pairs
    max_total = int(max_total)
    if max_total <= 0:
        logger.warning("Insufficient anchor pairs to enforce category balance")
        return positive_pairs
    targets: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for category, ratio in normalized_targets.items():
        desired = max_total * ratio
        take = min(int(desired), len(buckets[category]))
        targets[category] = take
        remainders.append((desired - take, category))
    allocated = sum(targets.values())
    remainder_slots = max_total - allocated
    remainders.sort(reverse=True)
    for remainder, category in remainders:
        if remainder_slots <= 0:
            break
        bucket = buckets[category]
        if targets[category] < len(bucket):
            targets[category] += 1
            remainder_slots -= 1
    selected: List[Tuple[Dict, Dict, str]] = []
    for category, take in targets.items():
        bucket = buckets[category]
        selected.extend(bucket[:take])
    if not selected:
        return positive_pairs
    anchor_counts = defaultdict(int)
    for anchor, _, _ in selected:
        anchor_counts[_chunk_category(anchor)] += 1
    total_selected = len(selected)
    summary_parts = []
    for cat in sorted(anchor_counts.keys()):
        count = anchor_counts[cat]
        pct = (count / total_selected) * 100 if total_selected else 0
        summary_parts.append(f"{cat}:{count} ({pct:.1f}%)")
    logger.info(
        "Anchor category mix enforced (down-sampled to %d pairs): %s",
        total_selected,
        ", ".join(summary_parts),
    )
    random.shuffle(selected)
    return selected


def _rebalance_same_source_pairs(
    positive_pairs: List[Tuple[Dict, Dict, str]]
) -> List[Tuple[Dict, Dict, str]]:
    """Ensure each anchor category keeps roughly the configured same-source ratio."""
    if not positive_pairs:
        return positive_pairs
    ratio = SAME_SOURCE_POSITIVE_RATIO
    if ratio <= 0 or ratio >= 1:
        return positive_pairs
    buckets: Dict[str, List[Tuple[Dict, Dict, str]]] = defaultdict(list)
    for pair in positive_pairs:
        buckets[_chunk_category(pair[0])].append(pair)
    adjusted: List[Tuple[Dict, Dict, str]] = []
    for category, cat_pairs in buckets.items():
        same_pairs = [p for p in cat_pairs if _chunk_category(p[0]) == _chunk_category(p[1])]
        cross_pairs = [p for p in cat_pairs if _chunk_category(p[0]) != _chunk_category(p[1])]
        random.shuffle(same_pairs)
        random.shuffle(cross_pairs)
        cat_total = len(cat_pairs)
        target_same = int(cat_total * ratio)
        selected_cat: List[Tuple[Dict, Dict, str]] = []
        same_take = min(len(same_pairs), target_same)
        selected_cat.extend(same_pairs[:same_take])
        remaining_slots = cat_total - len(selected_cat)
        cross_take = min(len(cross_pairs), remaining_slots)
        selected_cat.extend(cross_pairs[:cross_take])
        remaining_slots = cat_total - len(selected_cat)
        if remaining_slots > 0:
            leftovers = same_pairs[same_take:] + cross_pairs[cross_take:]
            random.shuffle(leftovers)
            selected_cat.extend(leftovers[:remaining_slots])
        adjusted.extend(selected_cat)
    random.shuffle(adjusted)
    if adjusted:
        same_count = sum(1 for pair in adjusted if _chunk_category(pair[0]) == _chunk_category(pair[1]))
        logger.info(
            "Positive pair mix enforced: %.1f%% same-source (target %.1f%%)",
            (same_count / len(adjusted)) * 100,
            ratio * 100,
        )
    return adjusted


def generate_negatives_by_difficulty(anchor: Dict, positive: Dict, 
                                   all_chunks: List[Dict], 
                                   domain_groups: Dict[str, List[Dict]]) -> Tuple[Dict, Dict, Dict]:
    """
    Generate easy, medium, and hard negatives for the same anchor/positive pair.
    
    EASY NEGATIVES: Random chunks from completely different domains
    - Different category entirely
    - Different file/document
    - Minimal topic overlap
    
    MEDIUM NEGATIVES: Chunks from similar topics but different contexts
    - Same general category but different subtopics
    - Different sections of documentation
    - Moderate semantic distance
    
    HARD NEGATIVES: Confusing chunks that look similar but aren't the answer
    - Same topic/category
    - Similar header paths or keywords
    - High semantic similarity but wrong context
    
    Returns: (easy_negative, medium_negative, hard_negative)
    """
    
    anchor_category = _chunk_category(anchor)
    anchor_title = anchor.get("title", "").lower()
    # Domain groups are keyed by category (and category+length) so mirror that here.
    anchor_domain = f"category::{anchor_category}"
    anchor_path = anchor.get("concat_header_path", "").lower()
    anchor_filename = anchor.get("filename", "")
    anchor_content_words = set(anchor.get("content", "").lower().split()[:30])
    
    # EASY NEGATIVE: Maximum distance - completely different domain
    easy_negative = _select_easy_negative(anchor, anchor_domain, domain_groups, all_chunks)
    
    # MEDIUM NEGATIVE: Moderate distance - similar domain, different context
    medium_negative = _select_medium_negative(anchor, anchor_path, anchor_filename, 
                                              anchor_content_words, all_chunks, positive)
    
    # HARD NEGATIVE: Minimum distance - same topic, confusingly similar
    hard_negative = _select_hard_negative(anchor, anchor_path, anchor_content_words, 
                                         all_chunks, positive, easy_negative, medium_negative)
    
    return easy_negative, medium_negative, hard_negative


def _select_easy_negative(anchor: Dict, anchor_domain: str, 
                         domain_groups: Dict[str, List[Dict]], 
                         all_chunks: List[Dict]) -> Dict:
    """Select an easy negative: different domain/topic and different H1 heading."""
    anchor_top_heading = _top_level_heading(anchor)

    def _sample_heading_safe(candidates: List[Dict]) -> Optional[Dict]:
        """Require negatives to avoid sharing the anchor's top-level heading."""
        filtered = [c for c in candidates if _top_level_heading(c) != anchor_top_heading]
        if filtered:
            return random.choice(filtered)
        return None

    # Try to find chunks from a different domain while guarding against heading collisions.
    if len(domain_groups) > 1:
        different_domains = [d for d in domain_groups.keys() 
                           if d != anchor_domain and not d.startswith("file_") 
                           and not d.startswith("topic_") and not d.startswith("short_")
                           and not d.startswith("medium_") and not d.startswith("long_")]
        
        if different_domains:
            neg_domain = random.choice(different_domains)
            candidate = _sample_heading_safe(domain_groups.get(neg_domain, []))
            if candidate:
                return candidate
    
    # Fallback: different filename, still enforcing the heading separation.
    different_file_chunks = [c for c in all_chunks 
                           if c.get("filename") != anchor.get("filename")]
    candidate = _sample_heading_safe(different_file_chunks)
    if candidate:
        return candidate
    
    # Ultimate fallback: any random chunk, again preferring different H1s when possible.
    other_chunks = [c for c in all_chunks if c.get("id") != anchor.get("id")]
    candidate = _sample_heading_safe(other_chunks)
    if candidate:
        return candidate

    # If no heading-safe negative exists, bail so the data bug is obvious.
    raise ValueError(
        "Unable to select easy negative with differing top-level heading for anchor "
        f"{anchor.get('id')}"
    )


def _select_medium_negative(anchor: Dict, anchor_path: str, anchor_filename: str,
                           anchor_content_words: Set[str], all_chunks: List[Dict],
                           positive: Dict) -> Dict:
    """Select a medium negative: same H1 as anchor but different context."""
    # Table rows can dominate certain headings, so we seed/update the table
    # cache up front and consult `_table_chunk_allowed_for_medium()` to cap how
    # many medium negatives can be tables under a single heading.
    _initialize_table_chunk_cache(all_chunks)
    anchor_top_heading = _top_level_heading(anchor)
    anchor_leaf = (anchor.get("heading") or anchor.get("concat_header_path") or "").lower()
    # Medium negatives stay under the same top-level heading so they remain on-topic,
    # but we bias toward different leaf sections to avoid trivial duplicates.
    preferred_candidates: List[Dict] = []  # Same H1 but different leaf section
    fallback_candidates: List[Dict] = []   # Same H1 even if leaf repeats
    
    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        # Skip anchor and positive
        if chunk_id == anchor.get("id") or chunk_id == positive.get("id"):
            continue

        chunk_top_heading = _top_level_heading(chunk)
        # Enforce same top-level heading for all medium negatives (hard constraint)
        if chunk_top_heading != anchor_top_heading:
            continue
        if not _table_chunk_allowed_for_medium(chunk, chunk_top_heading):
            continue
        
        chunk_path = chunk.get("concat_header_path", "").lower()
        chunk_filename = chunk.get("filename", "")
        
        # Look for chunks with some path similarity but not too much
        if anchor_path and chunk_path:
            anchor_parts = set(anchor_path.split("/"))
            chunk_parts = set(chunk_path.split("/"))
            overlap = len(anchor_parts.intersection(chunk_parts))
            total = len(anchor_parts.union(chunk_parts))
            path_similarity = overlap / total if total > 0 else 0
            
            # Medium difficulty: 20-50% path overlap
            if 0.2 <= path_similarity <= 0.5:
                target_list = preferred_candidates if _is_different_leaf(anchor_leaf, chunk) else fallback_candidates
                target_list.append(chunk)
            # Also accept same file but different section
            elif chunk_filename == anchor_filename and path_similarity < 0.3:
                target_list = preferred_candidates if _is_different_leaf(anchor_leaf, chunk) else fallback_candidates
                target_list.append(chunk)
    
    if preferred_candidates:
        return random.choice(preferred_candidates)
    if fallback_candidates:
        return random.choice(fallback_candidates)
    
    # Fallback: chunks with moderate content similarity within same H1
    for chunk in all_chunks:
        if chunk.get("id") in [anchor.get("id"), positive.get("id")]:
            continue
        chunk_top_heading = _top_level_heading(chunk)
        if chunk_top_heading != anchor_top_heading:
            continue
        if not _table_chunk_allowed_for_medium(chunk, chunk_top_heading):
            continue
        chunk_words = set(chunk.get("content", "").lower().split()[:30])
        if anchor_content_words and chunk_words:
            overlap = len(anchor_content_words.intersection(chunk_words))
            similarity = overlap / len(anchor_content_words.union(chunk_words))
            # 15-40% content similarity
            if 0.15 <= similarity <= 0.4:
                target_list = preferred_candidates if _is_different_leaf(anchor_leaf, chunk) else fallback_candidates
                target_list.append(chunk)

    if preferred_candidates:
        return random.choice(preferred_candidates)
    if fallback_candidates:
        return random.choice(fallback_candidates)
    
    # Ultimate fallback: any other chunk sharing the top-level heading
    other_chunks: List[Dict] = []
    for candidate in all_chunks:
        candidate_id = candidate.get("id")
        if candidate_id in [anchor.get("id"), positive.get("id")]:
            continue
        candidate_heading = _top_level_heading(candidate)
        if candidate_heading != anchor_top_heading:
            continue
        if not _table_chunk_allowed_for_medium(candidate, candidate_heading):
            continue
        other_chunks.append(candidate)
    if other_chunks:
        preferred = [c for c in other_chunks if _is_different_leaf(anchor_leaf, c)]
        if preferred:
            return random.choice(preferred)
        return random.choice(other_chunks)

    # If absolutely no same-heading negatives exist, fall back to original pool
    other_chunks = [
        c for c in all_chunks
        if c.get("id") not in [anchor.get("id"), positive.get("id")]
        and _table_chunk_allowed_for_medium(c, _top_level_heading(c))
    ]
    return random.choice(other_chunks) if other_chunks else all_chunks[0]


def _select_hard_negative(anchor: Dict, anchor_path: str, 
                         anchor_content_words: Set[str], all_chunks: List[Dict],
                         positive: Dict, easy_neg: Dict, medium_neg: Dict) -> Dict:
    """Select a hard negative: same topic, confusingly similar."""
    anchor_segments = _split_header_segments(anchor_path)
    if not anchor_segments:
        top_heading = _top_level_heading(anchor)
        if top_heading:
            anchor_segments = [top_heading]
    candidates = []
    anchor_category = _chunk_category(anchor)
    if not hasattr(_select_hard_negative, "_mix_stats"):
        _select_hard_negative._mix_stats = {"total": 0, "cross": 0}  # type: ignore[attr-defined]
    mix_stats = _select_hard_negative._mix_stats  # type: ignore[attr-defined]
    target_ratio = CROSS_SOURCE_HARD_NEGATIVE_RATIO
    current_ratio = (mix_stats["cross"] / mix_stats["total"]) if mix_stats["total"] else 0.0
    need_cross_source = target_ratio > 0 and current_ratio < target_ratio
    
    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        # Skip anchor, positive, and already selected negatives
        if chunk_id in [anchor.get("id"), positive.get("id"), easy_neg.get("id")]:
            continue
        
        chunk_path = chunk.get("concat_header_path", "").lower()
        chunk_heading = _top_level_heading(chunk)
        if not _shares_heading_scope(anchor_segments, chunk_path, chunk_heading):
            continue
        chunk_words = set(chunk.get("content", "").lower().split()[:30])
        chunk_category = _chunk_category(chunk)
        is_cross_source = chunk_category != anchor_category
        
        # Calculate similarities
        path_similarity = 0
        if anchor_path and chunk_path:
            anchor_parts = set(anchor_path.split("/"))
            chunk_parts = set(chunk_path.split("/"))
            overlap = len(anchor_parts.intersection(chunk_parts))
            total = len(anchor_parts.union(chunk_parts))
            path_similarity = overlap / total if total > 0 else 0
        
        content_similarity = 0
        if anchor_content_words and chunk_words:
            overlap = len(anchor_content_words.intersection(chunk_words))
            content_similarity = overlap / len(anchor_content_words.union(chunk_words))
        
        primary_match = (
            path_similarity >= HARD_PATH_SIM_THRESHOLD
            or content_similarity >= HARD_CONTENT_SIM_THRESHOLD
        )
        floor_match = (
            path_similarity >= HARD_MIN_PATH_FLOOR
            or content_similarity >= HARD_MIN_CONTENT_FLOOR
        )
        # Hard negative: keep the broader OR logic but ensure it passes a minimum similarity floor
        if primary_match and floor_match:
            # Score by combined similarity (higher is harder)
            score = path_similarity * 0.6 + content_similarity * 0.4
            candidates.append((chunk, score, is_cross_source))
    
    if candidates:
        # Sort by score and pick from top candidates (most confusing)
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = None
        if need_cross_source:
            cross_candidates = [c for c in candidates if c[2]]
            if cross_candidates:
                limit = min(HARD_CANDIDATE_SAMPLE_LIMIT, len(cross_candidates))
                top_cross = [c[0] for c in cross_candidates[:limit]]
                chosen = random.choice(top_cross)
        if not chosen:
            limit = min(HARD_CANDIDATE_SAMPLE_LIMIT, len(candidates))
            top_candidates = [c[0] for c in candidates[:limit]]
            chosen = random.choice(top_candidates)
        mix_stats["total"] += 1
        if _chunk_category(chosen) != anchor_category:
            mix_stats["cross"] += 1
        return chosen
    
    # Fallback: same file/category
    same_category_chunks = [
        c for c in all_chunks 
        if c.get("category") == anchor.get("category")
        and c.get("id") not in [anchor.get("id"), positive.get("id"),
                      easy_neg.get("id")]
        and _shares_heading_scope(anchor_segments, c.get("concat_header_path", "").lower(), _top_level_heading(c))
    ]
    if same_category_chunks:
        chosen = None
        if need_cross_source:
            cross_chunks = [c for c in same_category_chunks if _chunk_category(c) != anchor_category]
            if cross_chunks:
                chosen = random.choice(cross_chunks)
        if not chosen:
            chosen = random.choice(same_category_chunks)
        mix_stats["total"] += 1
        if _chunk_category(chosen) != anchor_category:
            mix_stats["cross"] += 1
        return chosen
    
    # Ultimate fallback: any different chunk
    other_chunks = [
        c for c in all_chunks 
        if c.get("id") not in [anchor.get("id"), positive.get("id"),
                     easy_neg.get("id")]
        and _shares_heading_scope(anchor_segments, c.get("concat_header_path", "").lower(), _top_level_heading(c))
    ]
    if other_chunks:
        chosen = None
        if need_cross_source:
            cross_chunks = [c for c in other_chunks if _chunk_category(c) != anchor_category]
            if cross_chunks:
                chosen = random.choice(cross_chunks)
        if not chosen:
            chosen = random.choice(other_chunks)
    else:
        fallback_pool = [
            c for c in all_chunks
            if c.get("id") not in [anchor.get("id"), positive.get("id"),
                                    easy_neg.get("id")]
        ]
        if fallback_pool:
            other_chunks = fallback_pool
            chosen = random.choice(other_chunks)
        else:
            chosen = all_chunks[0]
        mix_stats["total"] += 1
        if _chunk_category(chosen) != anchor_category:
            mix_stats["cross"] += 1
        return chosen
    mix_stats["total"] += 1
    if _chunk_category(chosen) != anchor_category:
        mix_stats["cross"] += 1
    return chosen


def generate_negative_pairs(positive_pairs: List[Tuple[Dict, Dict, str]], 
                          all_chunks: List[Dict], 
                          domain_groups: Dict[str, List[Dict]]) -> List[Tuple[Dict, Dict, str]]:
    """Generate negative pairs using multiple strategies for robust training."""
    negative_pairs = []
    
    for anchor, positive, pair_type in positive_pairs:
        negatives_found = 0
        
        # Strategy 1: Different domain (if multiple domains exist)
        if len(domain_groups) > 1:
            anchor_category = anchor.get("category", "unknown").lower()
            anchor_title = anchor.get("title", "").lower()
            anchor_domain = f"{anchor_category}_{anchor_title}" if anchor_title else anchor_category
            
            candidate_domains = [d for d in domain_groups.keys() if d != anchor_domain]
            if candidate_domains:
                for _ in range(min(NEGATIVE_SAMPLING_RATIO, len(candidate_domains))):
                    neg_domain = random.choice(candidate_domains)
                    if domain_groups[neg_domain]:
                        negative = random.choice(domain_groups[neg_domain])
                        negative_pairs.append((anchor, negative, f"negative_cross_domain_{pair_type}"))
                        negatives_found += 1
        
        # Strategy 2: Different filename/document  
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            different_file_chunks = [c for c in all_chunks 
                                   if c.get("filename") != anchor.get("filename")]
            if different_file_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(remaining_needed):
                    negative = random.choice(different_file_chunks)
                    negative_pairs.append((anchor, negative, f"negative_diff_file_{pair_type}"))
                    negatives_found += 1
        
        # Strategy 3: Different header path (semantic distance)
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            anchor_path = anchor.get("concat_header_path", "").lower()
            different_path_chunks = []
            
            for chunk in all_chunks:
                chunk_path = chunk.get("concat_header_path", "").lower()
                # Skip if same path or very similar path
                if chunk_path != anchor_path and chunk.get("id") != anchor.get("id"):
                    # Calculate simple path dissimilarity
                    if anchor_path and chunk_path:
                        anchor_parts = set(anchor_path.split("/"))
                        chunk_parts = set(chunk_path.split("/"))
                        overlap = len(anchor_parts.intersection(chunk_parts))
                        total = len(anchor_parts.union(chunk_parts))
                        similarity = overlap / total if total > 0 else 0
                        
                        # Use chunks with low path similarity as negatives
                        if similarity < 0.3:  # Less than 30% path overlap
                            different_path_chunks.append(chunk)
            
            if different_path_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(min(remaining_needed, len(different_path_chunks))):
                    negative = random.choice(different_path_chunks)
                    negative_pairs.append((anchor, negative, f"negative_diff_path_{pair_type}"))
                    negatives_found += 1
        
        # Strategy 4: Fallback - random sampling with content dissimilarity
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            anchor_words = set(anchor.get("content", "").lower().split()[:20])  # First 20 words
            dissimilar_chunks = []
            
            for chunk in all_chunks:
                if chunk.get("id") != anchor.get("id") and chunk.get("id") != positive.get("id"):
                    chunk_words = set(chunk.get("content", "").lower().split()[:20])
                    if anchor_words and chunk_words:
                        overlap = len(anchor_words.intersection(chunk_words))
                        similarity = overlap / len(anchor_words.union(chunk_words))
                        
                        # Use chunks with low content similarity
                        if similarity < 0.2:  # Less than 20% word overlap
                            dissimilar_chunks.append(chunk)
            
            if dissimilar_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(min(remaining_needed, len(dissimilar_chunks))):
                    negative = random.choice(dissimilar_chunks)
                    negative_pairs.append((anchor, negative, f"negative_dissimilar_{pair_type}"))
                    negatives_found += 1
        
        # Strategy 5: Ultimate fallback - completely random (avoid identical chunks)
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            available_chunks = [c for c in all_chunks 
                              if c.get("id") not in [anchor.get("id"), positive.get("id")]]
            if available_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(min(remaining_needed, len(available_chunks))):
                    negative = random.choice(available_chunks)
                    negative_pairs.append((anchor, negative, f"negative_random_{pair_type}"))
                    negatives_found += 1
    
    logger.info(f"Generated {len(negative_pairs)} negative pairs using multiple strategies",
                extra={"Triplets Generated": str(len(negative_pairs)), "Training Data Type": "negative_pairs"})
    return negative_pairs


def create_training_triplets_by_difficulty(positive_pairs: List[Tuple[Dict, Dict, str]], 
                                          all_chunks: List[Dict],
                                          domain_groups: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Create training triplets with easy, medium, and hard negatives for curriculum learning.
    
    Returns dictionary with keys: 'easy', 'medium', 'hard'
    Each contains list of triplets with appropriate difficulty negatives.
    """
    triplets_by_difficulty = {
        'easy': [],
        'medium': [],
        'hard': []
    }
    
    logger.info(f"Generating triplets with difficulty levels for {len(positive_pairs)} positive pairs")
    hard_total = 0
    hard_cross_source = 0
    discarded_triplets = {"easy": 0, "medium": 0, "hard": 0}
    candidate_counts = {"easy": 0, "medium": 0, "hard": 0}
    
    # DIFFICULTY FILTER ORDER: generate candidate negatives (rule-based per
    # difficulty), then immediately run cosine threshold checks so any
    # over-similar triplets are discarded before export/stat tracking.
    for anchor, positive, pos_type in positive_pairs:
        # Generate all three difficulty levels for this pair
        easy_neg, medium_neg, hard_neg = generate_negatives_by_difficulty(
            anchor, positive, all_chunks, domain_groups
        )
        
        anchor_category = _chunk_category(anchor)
        positive_category = _chunk_category(positive)
        easy_category = _chunk_category(easy_neg)
        medium_category = _chunk_category(medium_neg)
        hard_category = _chunk_category(hard_neg)

        # EASY FILTERS: drop tables (handled via exclusions), require a distinct
        # top-level heading, and enforce the low-cosine threshold so negatives
        # feel obviously wrong to the model at the start of training.
        candidate_counts["easy"] += 1
        too_similar, easy_cos = _negative_similarity_exceeds_threshold(anchor, easy_neg, "easy")
        if too_similar:
            discarded_triplets["easy"] += 1
            if easy_cos is not None:
                logger.debug(
                    "Discarded easy triplet for anchor %s (cos=%.3f > %.2f)",
                    anchor.get("id"),
                    easy_cos,
                    EASY_NEGATIVE_MAX_COSINE,
                )
        else:
            easy_triplet = {
                "anchor": _extract_text_for_training(anchor, include_instruction=True),
                "positive": _extract_text_for_training(positive),
                "negative": _extract_text_for_training(easy_neg),
                "anchor_id": anchor.get("id"),
                "positive_id": positive.get("id"),
                "negative_id": easy_neg.get("id"),
                "pair_type": pos_type,
                "difficulty": "easy",
                "negative_type": "cross_domain",
                "anchor_category": anchor_category,
                "positive_category": positive_category,
                "negative_category": easy_category,
                "anchor_domain": f"{anchor.get('category', '')}_{anchor.get('title', '')}",
                "positive_domain": f"{positive.get('category', '')}_{positive.get('title', '')}",
                "negative_domain": f"{easy_neg.get('category', '')}_{easy_neg.get('title', '')}"
            }
            triplets_by_difficulty['easy'].append(easy_triplet)
        
        # MEDIUM FILTERS: allow same H1 but demand different leaf/context and a
        # cosine window (min/max). Table throttling ensures we do not oversample
        # structured data for this tier.
        candidate_counts["medium"] += 1
        too_similar, medium_cos = _negative_similarity_exceeds_threshold(anchor, medium_neg, "medium")
        if too_similar:
            discarded_triplets["medium"] += 1
            if medium_cos is not None:
                if MEDIUM_NEGATIVE_MIN_COSINE is not None and medium_cos < MEDIUM_NEGATIVE_MIN_COSINE:
                    logger.debug(
                        "Discarded medium triplet for anchor %s (cos=%.3f < %.2f)",
                        anchor.get("id"),
                        medium_cos,
                        MEDIUM_NEGATIVE_MIN_COSINE,
                    )
                else:
                    logger.debug(
                        "Discarded medium triplet for anchor %s (cos=%.3f > %.2f)",
                        anchor.get("id"),
                        medium_cos,
                        MEDIUM_NEGATIVE_MAX_COSINE,
                    )
        else:
            medium_triplet = {
                "anchor": _extract_text_for_training(anchor, include_instruction=True),
                "positive": _extract_text_for_training(positive),
                "negative": _extract_text_for_training(medium_neg),
                "anchor_id": anchor.get("id"),
                "positive_id": positive.get("id"),
                "negative_id": medium_neg.get("id"),
                "pair_type": pos_type,
                "difficulty": "medium",
                "negative_type": "similar_topic",
                "anchor_category": anchor_category,
                "positive_category": positive_category,
                "negative_category": medium_category,
                "anchor_domain": f"{anchor.get('category', '')}_{anchor.get('title', '')}",
                "positive_domain": f"{positive.get('category', '')}_{positive.get('title', '')}",
                "negative_domain": f"{medium_neg.get('category', '')}_{medium_neg.get('title', '')}"
            }
            triplets_by_difficulty['medium'].append(medium_triplet)
        
        # HARD FILTERS: enforce shared topic scope plus the highest cosine band
        # (still below the hard max) so negatives are “confusingly similar”.
        candidate_counts["hard"] += 1
        too_similar, hard_cos = _negative_similarity_exceeds_threshold(anchor, hard_neg, "hard")
        if too_similar:
            discarded_triplets["hard"] += 1
            if hard_cos is not None:
                logger.debug(
                    "Discarded hard triplet for anchor %s (cos=%.3f > %.2f)",
                    anchor.get("id"),
                    hard_cos,
                    HARD_NEGATIVE_MAX_COSINE,
                )
        else:
            hard_triplet = {
                "anchor": _extract_text_for_training(anchor, include_instruction=True),
                "positive": _extract_text_for_training(positive),
                "negative": _extract_text_for_training(hard_neg),
                "anchor_id": anchor.get("id"),
                "positive_id": positive.get("id"),
                "negative_id": hard_neg.get("id"),
                "pair_type": pos_type,
                "difficulty": "hard",
                "negative_type": "confusingly_similar",
                "anchor_category": anchor_category,
                "positive_category": positive_category,
                "negative_category": hard_category,
                "anchor_domain": f"{anchor.get('category', '')}_{anchor.get('title', '')}",
                "positive_domain": f"{positive.get('category', '')}_{positive.get('title', '')}",
                "negative_domain": f"{hard_neg.get('category', '')}_{hard_neg.get('title', '')}"
            }
            triplets_by_difficulty['hard'].append(hard_triplet)
            hard_total += 1
            if hard_category != anchor_category:
                hard_cross_source += 1
    
    for difficulty, triplets in triplets_by_difficulty.items():
        logger.info(f"Created {len(triplets)} {difficulty} triplets",
                   extra={"Triplets Generated": str(len(triplets)), 
                         "Training Data Type": f"{difficulty}_triplets"})
    if hard_total:
        logger.info(
            "Hard negative mix: %.1f%% cross-source (target %.1f%%)",
            (hard_cross_source / hard_total) * 100,
            CROSS_SOURCE_HARD_NEGATIVE_RATIO * 100,
        )
    
    for difficulty in ("easy", "medium", "hard"):
        found = candidate_counts[difficulty]
        removed = discarded_triplets[difficulty]
        logger.info(
            "%s triplets: Found %d, removed %d due to exceeding cosine similarity threshold",
            difficulty.capitalize(),
            found,
            removed,
        )

    triplets_by_difficulty = _enforce_triplet_mix(triplets_by_difficulty)
    return triplets_by_difficulty, candidate_counts, discarded_triplets


def _enforce_triplet_mix(triplets_by_difficulty: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Down-sample difficulty buckets to match the configured triplet mix."""
    ratios = {
        "easy": EASY_TRIPLET_RATIO,
        "medium": MEDIUM_TRIPLET_RATIO,
        "hard": HARD_TRIPLET_RATIO,
    }
    active_ratios = {k: v for k, v in ratios.items() if v > 0}
    if not active_ratios:
        return triplets_by_difficulty
    missing = [k for k in active_ratios if not triplets_by_difficulty.get(k)]
    if missing:
        logger.warning(
            "Cannot enforce triplet mix; no triplets generated for: %s",
            ", ".join(sorted(missing)),
        )
        return triplets_by_difficulty
    ratio_sum = sum(active_ratios.values())
    if ratio_sum <= 0:
        return triplets_by_difficulty
    normalized = {k: v / ratio_sum for k, v in active_ratios.items()}
    max_total = min(len(triplets_by_difficulty[k]) / normalized[k] for k in normalized)
    max_total = int(max_total)
    if max_total <= 0:
        logger.warning("Triplet mix enforcement would remove all samples; skipping rebalancing")
        return triplets_by_difficulty
    targets = {}
    remainders = []
    for difficulty, ratio in normalized.items():
        desired = max_total * ratio
        target = min(len(triplets_by_difficulty[difficulty]), int(desired))
        targets[difficulty] = target
        remainders.append((desired - target, difficulty))
    allocated = sum(targets.values())
    remainders.sort(reverse=True)
    for remainder, difficulty in remainders:
        if allocated >= max_total:
            break
        if targets[difficulty] < len(triplets_by_difficulty[difficulty]):
            targets[difficulty] += 1
            allocated += 1
    for difficulty, target in targets.items():
        if len(triplets_by_difficulty[difficulty]) > target:
            random.shuffle(triplets_by_difficulty[difficulty])
            triplets_by_difficulty[difficulty] = triplets_by_difficulty[difficulty][:target]
    logger.info(
        "Applied triplet mix ratios (easy %.0f%% / medium %.0f%% / hard %.0f%%) => counts: easy %d, medium %d, hard %d (total %d)",
        EASY_TRIPLET_RATIO * 100,
        MEDIUM_TRIPLET_RATIO * 100,
        HARD_TRIPLET_RATIO * 100,
        len(triplets_by_difficulty.get("easy", [])),
        len(triplets_by_difficulty.get("medium", [])),
        len(triplets_by_difficulty.get("hard", [])),
        sum(len(v) for v in triplets_by_difficulty.values()),
    )
    return triplets_by_difficulty


def _extract_text_for_training(chunk: Dict, include_instruction: bool = False) -> str:
    """Extract the best text representation, optionally prefixing the embed instruction."""
    content = chunk.get("content", "").strip()
    summary = chunk.get("chunk_summary", "").strip()
    heading = chunk.get("concat_header_path", "").strip()
    
    # Prefer summary if available and not a placeholder
    if summary and summary.lower() != "false" and len(summary) > 20:
        if heading:
            text = f"{heading}: {summary}"
        else:
            text = summary

    # Fallback to content with optional heading
    elif heading and len(content) > 100:
        text = f"{heading}: {content}"

    else:
        text = content

    if include_instruction:
        return f"{EMBED_INSTRUCTION}\n{text}"

    return text

def export_training_data_by_difficulty(
    triplets_by_difficulty: Dict[str, List[Dict]],
    base_output_dir: Path,
    candidate_counts: Optional[Dict[str, int]] = None,
    discarded_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Export training data separated by difficulty level for curriculum learning.
    
    Creates subdirectories: easy/, medium/, hard/
    Each contains train/test splits in JSON format for the orchestrator to use.
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    for difficulty, triplets in triplets_by_difficulty.items():
        if not triplets:
            logger.warning(f"No {difficulty} triplets to export")
            continue
        
        # Create difficulty-specific subdirectory
        difficulty_dir = base_output_dir / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/test
        random.shuffle(triplets)
        split_idx = int(len(triplets) * TRAIN_TEST_SPLIT)
        train_triplets = triplets[:split_idx]
        test_triplets = triplets[split_idx:]
        
        # Export in JSON format (for tokenizer compatibility)
        _export_json_format(train_triplets, difficulty_dir / "triplets_train.json")
        _export_json_format(test_triplets, difficulty_dir / "triplets_test.json")
        
        # Export statistics
        extra_stats = {
            "triplets_found": (candidate_counts or {}).get(difficulty, 0),
            "triplets_removed": (discarded_counts or {}).get(difficulty, 0),
        }
        _export_statistics(triplets, difficulty_dir / "statistics.json", extra_stats)
        
        logger.info(f"Exported {difficulty} data to {difficulty_dir}: "
                   f"{len(train_triplets)} train, {len(test_triplets)} test",
                   extra={"Triplets Generated": str(len(triplets)), 
                         "Training Data Type": f"{difficulty}_export"})


def _export_json_format(triplets: List[Dict], output_path: Path) -> None:
    """Export triplets in JSON format for the training pipeline."""
    # Simplify triplets to just the essential fields for training
    simplified = []
    for t in triplets:
        simplified.append({
            "anchor": t["anchor"],
            "positive": t["positive"],
            "negative": t["negative"],
            "difficulty": t.get("difficulty", "unknown"),
            "pair_type": t.get("pair_type", "unknown")
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)


def _export_statistics(triplets: List[Dict], output_path: Path, extra_stats: Optional[Dict[str, int]] = None) -> None:
    """Export training data statistics."""
    stats = {
        "total_triplets": len(triplets),
        "difficulty": triplets[0].get("difficulty", "unknown") if triplets else "unknown",
        "pair_types": {},
        "domains": {},
        "avg_text_lengths": {}
    }

    if extra_stats:
        stats.update({k: int(v) for k, v in extra_stats.items() if isinstance(v, (int, float))})
    
    # Analyze pair types
    for triplet in triplets:
        pair_type = triplet.get("pair_type", "unknown")
        stats["pair_types"][pair_type] = stats["pair_types"].get(pair_type, 0) + 1
    
    # Analyze domains
    for triplet in triplets:
        domain = triplet.get("anchor_domain", "unknown")
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
    
    # Analyze text lengths
    anchor_lengths = [len(t["anchor"]) for t in triplets]
    positive_lengths = [len(t["positive"]) for t in triplets]
    negative_lengths = [len(t["negative"]) for t in triplets]
    
    stats["avg_text_lengths"] = {
        "anchor": sum(anchor_lengths) / len(anchor_lengths) if anchor_lengths else 0,
        "positive": sum(positive_lengths) / len(positive_lengths) if positive_lengths else 0,
        "negative": sum(negative_lengths) / len(negative_lengths) if negative_lengths else 0
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def main():
    """Main function to generate training data for embedding model fine-tuning."""
    logger.info("Starting embedding model training data generation with difficulty levels")
    
    # Load and filter chunks
    chunks, metadata = load_chunks()
    filtered_chunks = filter_chunks_for_training(chunks)
    _ensure_chunk_embeddings(filtered_chunks)
    
    if len(filtered_chunks) < 100:
        logger.warning(f"Only {len(filtered_chunks)} chunks available - may not be sufficient for training")
    
    # Group by domain for systematic sampling
    domain_groups = group_chunks_by_domain(filtered_chunks)
    
    # Generate positive pairs
    positive_pairs = generate_positive_pairs(filtered_chunks)
    
    if not positive_pairs:
        logger.error("No positive pairs generated - cannot create training data")
        return
    
    logger.info(f"Generated {len(positive_pairs)} positive pairs")
    
    # Create triplets with easy, medium, and hard negatives for curriculum learning
    triplets_by_difficulty, candidate_counts, discarded_triplets = create_training_triplets_by_difficulty(
        positive_pairs, filtered_chunks, domain_groups
    )
    
    total_triplets = sum(len(t) for t in triplets_by_difficulty.values())
    if total_triplets == 0:
        logger.error("No triplets created - check positive/negative pair generation")
        return
    
    # Export training data by difficulty level
    output_dir = TRAINING_DATA_DIR
    export_training_data_by_difficulty(triplets_by_difficulty, output_dir, candidate_counts, discarded_triplets)
    
    logger.info(f"Training data generation complete:")
    logger.info(f"  Easy: {len(triplets_by_difficulty['easy'])} triplets")
    logger.info(f"  Medium: {len(triplets_by_difficulty['medium'])} triplets")
    logger.info(f"  Hard: {len(triplets_by_difficulty['hard'])} triplets")
    logger.info(f"  Total: {total_triplets} triplets exported to {output_dir}")
    logger.info(f"Data ready for curriculum learning with run_multi_epoch_training.py")


if __name__ == "__main__":
    main()
