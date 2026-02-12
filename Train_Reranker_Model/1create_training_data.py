

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from itertools import combinations

from scripts.config_training_rerank import *
from scripts.custom_logger import setup_global_logger

# High-level flow::
# 1. Load the chunk JSON file that was generated earlier in the pipeline.
# 2. Generate training triplets (anchor, positive, negative) from the chunks:
#    - Use hierarchical relationships (parent-child) for positive pairs
#    - Use semantic similarity (same heading paths, summaries) for positive pairs
#    - Generate THREE difficulty levels of negatives for each positive pair:
#      * EASY: Random chunks from completely different domains/topics
#      * MEDIUM: Chunks from similar topics but different contexts
#      * HARD: Chunks from same topic that look similar but aren't the answer
# 3. Export training data separated by difficulty into subdirectories:
#    - 
#    - reranker_training_data/easy/  (for epoch 1 training)
#    - reranker_training_data/medium/  (for epoch 2 training)
#    - reranker_training_data/hard/  (for epoch 3 training)
# 4. Each difficulty level gets train/test splits for evaluation
# 5. Data is ready for curriculum learning with run_multi_epoch_training.py
#
# WHY DIFFICULTY LEVELS:
# - Curriculum learning: Train on easy examples first, then progressively harder
# - Easy negatives teach basic domain separation
# - Medium negatives teach topic-level distinctions
# - Hard negatives teach fine-grained semantic differences
# - This progressive approach often produces better final models than random negatives

DEFAULT_CHUNK_FILE = BASE_CWD / "a_chunks.json"

# Set up global logger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Triplets Generated", "Training Data Type"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)


def _chunk_category(chunk: Dict) -> str:
    """Return a normalized category label for a chunk."""
    return (chunk.get("category") or "unknown").lower()


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


def load_chunks(chunks_path: Path = DEFAULT_CHUNK_FILE) -> Tuple[List[Dict], Dict]:
    """Load chunks from JSON file and return chunks list plus metadata."""
    resolved_path = Path(chunks_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    
    if isinstance(loaded, dict) and "chunks" in loaded:
        chunks = loaded["chunks"]
        metadata = {k: v for k, v in loaded.items() if k != "chunks"}
    else:
        chunks = loaded
        metadata = {}
    
    logger.info(f"Loaded {len(chunks)} chunks from {resolved_path}")
    return chunks, metadata


def filter_chunks_for_training(chunks: List[Dict]) -> List[Dict]:
    """Filter chunks suitable for training based on content length and quality."""
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
    
    logger.info(f"Generated {len(positive_pairs)} positive pairs", 
                extra={"Triplets Generated": str(len(positive_pairs)), "Training Data Type": "positive_pairs"})
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
    """Select an easy negative: completely different domain/topic."""
    # Try to find chunks from a different domain
    if len(domain_groups) > 1:
        different_domains = [d for d in domain_groups.keys() 
                           if d != anchor_domain and not d.startswith("file_") 
                           and not d.startswith("topic_") and not d.startswith("short_")
                           and not d.startswith("medium_") and not d.startswith("long_")]
        
        if different_domains:
            neg_domain = random.choice(different_domains)
            if domain_groups[neg_domain]:
                return random.choice(domain_groups[neg_domain])
    
    # Fallback: different filename
    different_file_chunks = [c for c in all_chunks 
                           if c.get("filename") != anchor.get("filename")]
    if different_file_chunks:
        return random.choice(different_file_chunks)
    
    # Ultimate fallback: any random chunk
    other_chunks = [c for c in all_chunks if c.get("id") != anchor.get("id")]
    return random.choice(other_chunks) if other_chunks else all_chunks[0]


def _select_medium_negative(anchor: Dict, anchor_path: str, anchor_filename: str,
                           anchor_content_words: Set[str], all_chunks: List[Dict],
                           positive: Dict) -> Dict:
    """Select a medium negative: similar topic but different context."""
    candidates = []
    
    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        # Skip anchor and positive
        if chunk_id == anchor.get("id") or chunk_id == positive.get("id"):
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
                candidates.append(chunk)
            # Also accept same file but different section
            elif chunk_filename == anchor_filename and path_similarity < 0.3:
                candidates.append(chunk)
    
    if candidates:
        return random.choice(candidates)
    
    # Fallback: chunks with moderate content similarity
    moderate_similarity_chunks = []
    for chunk in all_chunks:
        if chunk.get("id") not in [anchor.get("id"), positive.get("id")]:
            chunk_words = set(chunk.get("content", "").lower().split()[:30])
            if anchor_content_words and chunk_words:
                overlap = len(anchor_content_words.intersection(chunk_words))
                similarity = overlap / len(anchor_content_words.union(chunk_words))
                # 15-40% content similarity
                if 0.15 <= similarity <= 0.4:
                    moderate_similarity_chunks.append(chunk)
    
    if moderate_similarity_chunks:
        return random.choice(moderate_similarity_chunks)
    
    # Ultimate fallback: any different chunk
    other_chunks = [c for c in all_chunks 
                   if c.get("id") not in [anchor.get("id"), positive.get("id")]]
    return random.choice(other_chunks) if other_chunks else all_chunks[0]


def _select_hard_negative(anchor: Dict, anchor_path: str, 
                         anchor_content_words: Set[str], all_chunks: List[Dict],
                         positive: Dict, easy_neg: Dict, medium_neg: Dict) -> Dict:
    """Select a hard negative: same topic, confusingly similar."""
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
        if chunk_id in [anchor.get("id"), positive.get("id"), 
                       easy_neg.get("id"), medium_neg.get("id")]:
            continue
        
        chunk_path = chunk.get("concat_header_path", "").lower()
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
        
        # Hard negative: high path similarity (>50%) OR high content similarity (>40%)
        if path_similarity > 0.5 or content_similarity > 0.4:
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
                top_cross = [c[0] for c in cross_candidates[: min(5, len(cross_candidates))]]
                chosen = random.choice(top_cross)
        if not chosen:
            top_candidates = [c[0] for c in candidates[: min(5, len(candidates))]]
            chosen = random.choice(top_candidates)
        mix_stats["total"] += 1
        if _chunk_category(chosen) != anchor_category:
            mix_stats["cross"] += 1
        return chosen
    
    # Fallback: same file/category
    same_category_chunks = [c for c in all_chunks 
                           if c.get("category") == anchor.get("category")
                           and c.get("id") not in [anchor.get("id"), positive.get("id"),
                                                   easy_neg.get("id"), medium_neg.get("id")]]
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
    other_chunks = [c for c in all_chunks 
                   if c.get("id") not in [anchor.get("id"), positive.get("id"),
                                         easy_neg.get("id"), medium_neg.get("id")]]
    if other_chunks:
        chosen = None
        if need_cross_source:
            cross_chunks = [c for c in other_chunks if _chunk_category(c) != anchor_category]
            if cross_chunks:
                chosen = random.choice(cross_chunks)
        if not chosen:
            chosen = random.choice(other_chunks)
    else:
        chosen = all_chunks[0]
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
    candidate_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    discarded_triplets = {'easy': 0, 'medium': 0, 'hard': 0}
    
    logger.info(f"Generating triplets with difficulty levels for {len(positive_pairs)} positive pairs")
    hard_total = 0
    hard_cross_source = 0
    
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

        # Create easy triplet
        easy_triplet = {
            "anchor": _extract_text_for_training(anchor),
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
        candidate_counts['easy'] += 1
        
        # Create medium triplet
        medium_triplet = {
            "anchor": _extract_text_for_training(anchor),
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
        candidate_counts['medium'] += 1
        
        # Create hard triplet
        hard_triplet = {
            "anchor": _extract_text_for_training(anchor),
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
        candidate_counts['hard'] += 1
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
    
    pre_enforce_counts = {k: len(v) for k, v in triplets_by_difficulty.items()}
    triplets_by_difficulty = _enforce_triplet_mix(triplets_by_difficulty)
    for difficulty, before in pre_enforce_counts.items():
        after = len(triplets_by_difficulty.get(difficulty, []))
        discarded_triplets[difficulty] += max(0, before - after)

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


def _extract_text_for_training(chunk: Dict) -> str:
    """Extract the best text representation for training from a chunk."""
    content = chunk.get("content", "").strip()
    summary = chunk.get("chunk_summary", "").strip()
    heading = chunk.get("concat_header_path", "").strip()
    
    # Prefer summary if available and not a placeholder
    if summary and summary.lower() != "false" and len(summary) > 20:
        if heading:
            return f"{heading}: {summary}"
        return summary
    
    # Fallback to content with optional heading
    if heading and len(content) > 100:
        return f"{heading}: {content}"
    
    return content


def export_training_data_by_difficulty(
    triplets_by_difficulty: Dict[str, List[Dict]],
    base_output_dir: Path,
    candidate_counts: Optional[Dict[str, int]] = None,
    discarded_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Export triplets split by difficulty + train/test and return the in-memory splits.

    Returning the split data lets us immediately derive cross-encoder pairs without re-reading
    from disk or attempting to reproduce the exact shuffle boundary again.
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    split_cache: Dict[str, Dict[str, List[Dict]]] = {}

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
        split_cache[difficulty] = {
            "train": list(train_triplets),
            "test": list(test_triplets),
        }

        # Export in JSON format (for tokenizer compatibility)
        _export_json_format(train_triplets, difficulty_dir / "triplets_train.json")
        _export_json_format(test_triplets, difficulty_dir / "triplets_test.json")

        # Export statistics
        extra_stats = {
            "triplets_found": (candidate_counts or {}).get(difficulty, 0),
            "triplets_removed": (discarded_counts or {}).get(difficulty, 0),
        }
        _export_statistics(triplets, difficulty_dir / "statistics.json", extra_stats)

        logger.info(
            f"Exported {difficulty} data to {difficulty_dir}: "
            f"{len(train_triplets)} train, {len(test_triplets)} test",
            extra={
                "Triplets Generated": str(len(triplets)),
                "Training Data Type": f"{difficulty}_export",
            },
        )

    return split_cache


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


def load_existing_triplet_splits(data_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """Load pre-generated triplet splits from disk for cross-encoder export."""
    difficulties = ["easy", "medium", "hard"]
    split_triplets: Dict[str, Dict[str, List[Dict]]] = {}
    missing: List[str] = []

    for difficulty in difficulties:
        diff_dir = data_dir / difficulty
        train_path = diff_dir / "triplets_train.json"
        test_path = diff_dir / "triplets_test.json"
        if not train_path.exists() or not test_path.exists():
            missing.append(difficulty)
            continue

        with open(train_path, "r", encoding="utf-8") as f_train:
            train_triplets = json.load(f_train)
        with open(test_path, "r", encoding="utf-8") as f_test:
            test_triplets = json.load(f_test)

        split_triplets[difficulty] = {
            "train": train_triplets,
            "test": test_triplets,
        }

    if not split_triplets:
        raise FileNotFoundError(
            f"No triplet splits found under {data_dir}; run with --regenerate-triplets first."
        )

    if missing:
        logger.warning(
            "Missing triplet files for difficulties: %s. Cross-encoder export will skip them.",
            ", ".join(missing),
        )

    total_records = sum(
        len(splits.get("train", [])) + len(splits.get("test", []))
        for splits in split_triplets.values()
    )
    logger.info(
        "Loaded %d triplet records from %s for cross-encoder export",
        total_records,
        data_dir,
    )
    return split_triplets


def export_cross_encoder_pairs(
    split_triplets: Dict[str, Dict[str, List[Dict]]],
    pair_output_dir: Path,
) -> None:
    """Create cross-encoder (query, candidate, label) pairs from the triplet splits.

    Each triplet produces two supervised pairs: (anchor, positive, label=1) and
    (anchor, negative, label=0). We persist the results as newline-delimited JSON so they can be
    streamed efficiently by the cross-encoder trainer without loading everything into memory.
    """

    pair_output_dir.mkdir(parents=True, exist_ok=True)
    pair_counts: Dict[str, Dict[str, int]] = {}

    for difficulty, splits in split_triplets.items():
        diff_dir = pair_output_dir / difficulty
        diff_dir.mkdir(parents=True, exist_ok=True)
        pair_counts[difficulty] = {}

        for split_name, triplets in splits.items():
            if not triplets:
                logger.warning("No %s triplets found for %s split; skipping cross-encoder export", difficulty, split_name)
                continue

            out_path = diff_dir / f"{split_name}.jsonl"
            pairs_written = 0
            with open(out_path, "w", encoding="utf-8") as handle:
                for idx, triplet in enumerate(triplets):
                    base_record = {
                        "query": triplet["anchor"],
                        "difficulty": difficulty,
                        "pair_type": triplet.get("pair_type", "unknown"),
                        "triplet_index": idx,
                    }
                    # Positive candidate
                    pos_record = dict(base_record)
                    pos_record.update({
                        "candidate": triplet["positive"],
                        "label": 1,
                        "candidate_role": "positive",
                    })
                    handle.write(json.dumps(pos_record, ensure_ascii=False) + "\n")

                    neg_record = dict(base_record)
                    neg_record.update({
                        "candidate": triplet["negative"],
                        "label": 0,
                        "candidate_role": "negative",
                    })
                    handle.write(json.dumps(neg_record, ensure_ascii=False) + "\n")
                    pairs_written += 2

            pair_counts[difficulty][split_name] = pairs_written
            logger.info(
                "Exported %d cross-encoder pairs for %s/%s to %s",
                pairs_written,
                difficulty,
                split_name,
                out_path,
            )

    metadata = {
        "pair_counts": pair_counts,
        "total_pairs": sum(sum(split.values()) for split in pair_counts.values()),
        "schema": {
            "query": "User question / anchor text",
            "candidate": "Candidate passage to score",
            "label": "1 if relevant, 0 otherwise",
            "difficulty": "Difficulty bucket inherited from triplet",
        },
    }
    with open(pair_output_dir / "metadata.json", "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for triplet regeneration and overrides."""
    parser = argparse.ArgumentParser(
        description=(
            "Create reranker triplets from chunked data or reuse the merged "
            "embedding outputs to refresh cross-encoder pairs."
        )
    )
    parser.add_argument(
        "--regenerate-triplets",
        action="store_true",
        help=(
            "Rebuild triplets from chunk JSON instead of reusing the merged "
            "embedding data under reranker_training_data."
        ),
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=DEFAULT_CHUNK_FILE,
        help="Path to the chunk JSON to use when regenerating triplets.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Generate or reuse triplets before exporting cross-encoder supervision pairs."""
    if args is None:
        args = parse_args()

    if args.regenerate_triplets:
        chunks, _ = load_chunks(args.chunks_file)
        filtered_chunks = filter_chunks_for_training(chunks)
        if len(filtered_chunks) < 100:
            logger.warning(
                "Only %d filtered chunks available; curriculum coverage may suffer.",
                len(filtered_chunks),
            )

        domain_groups = group_chunks_by_domain(filtered_chunks)
        positive_pairs = generate_positive_pairs(filtered_chunks)
        if not positive_pairs:
            logger.error("No positive pairs generated - cannot create training data")
            return

        logger.info("Generated %d positive pairs", len(positive_pairs))
        triplets_by_difficulty, candidate_counts, discarded_triplets = create_training_triplets_by_difficulty(
            positive_pairs,
            filtered_chunks,
            domain_groups,
        )
        total_triplets = sum(len(t) for t in triplets_by_difficulty.values())
        if total_triplets == 0:
            logger.error("No triplets created - check positive/negative pair generation")
            return

        split_triplets = export_training_data_by_difficulty(
            triplets_by_difficulty,
            TRAINING_DATA_DIR,
            candidate_counts,
            discarded_triplets,
        )
    else:
        logger.info(
            "Reusing merged embedding triplets from %s (sources: %s)",
            TRAINING_DATA_DIR,
            ", ".join(EMBED_SOURCE_SUBDIRS),
        )
        split_triplets = load_existing_triplet_splits(TRAINING_DATA_DIR)

    export_cross_encoder_pairs(split_triplets, CROSS_ENCODER_DATA_DIR)

    triplet_counts = {
        difficulty: sum(len(records) for records in splits.values())
        for difficulty, splits in split_triplets.items()
    }
    total_triplets = sum(triplet_counts.values())

    logger.info("Training triplet summary:")
    for difficulty in sorted(triplet_counts.keys()):
        logger.info("  %s: %d triplets", difficulty.title(), triplet_counts[difficulty])
    logger.info("  Total: %d triplets available at %s", total_triplets, TRAINING_DATA_DIR)
    logger.info("Cross-encoder pairs ready at %s", CROSS_ENCODER_DATA_DIR)
    logger.info("Data ready for curriculum learning + reranker fine-tuning")


if __name__ == "__main__":
    main()
