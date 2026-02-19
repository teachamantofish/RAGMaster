"""Exp 33: Apply manual labels from label_candidates.md to retriever_eval_queries.json.

Reads the user's picks, updates each excluded query:
- Queries with chunk IDs -> split="train", add positive_ids, remove positive_filters
- Queries marked "delete this question" -> removed entirely

Also fixes garbled q015 entry.
"""

import json
from pathlib import Path

JSON_PATH = Path(__file__).parent / "retriever_eval_queries.json"

# ── User picks parsed from label_candidates.md ──
# Maps query_id -> list of positive chunk IDs (or "DELETE")
PICKS = {
    "q001": ["h3_eb897618_tab", "h3_fcc1c11d_tab"],
    "q002": ["h2_be9735b6"],
    "q003": ["h2_a9464f08", "h2_cdb1ca35_exa"],
    "q004": ["h2_27595b92", "h2_96cc0c95_exa"],
    "q005": ["h2_27595b92", "h2_43d20e69"],
    "q006": ["h3_fcc1c11d_tab", "h3_d1bb5751"],
    "q007": ["h3_54614d1a", "h3_5f87d866"],
    "q008": ["h3_701e0ac0", "h4_1d322d63", "h4_d496da3d"],
    "q009": ["h4_35683d09"],
    "q010": ["h2_be6149de_tab", "h3_dab46513"],
    "q011": ["h3_0c7d6f50", "h4_bbdbd7bc_tab"],
    "q012": ["h4_bc9bd4ea_tab", "h3_6122a9d9"],
    "q013": "DELETE",
    "q014": ["h2_6ad963d1", "h2_d46584ad"],
    "q015": ["h2_42620f51_tab", "h2_a97f3127_tab", "h3_85ebc697_tab"],  # fixed garbled entry
    "q016": ["h3_be4b9cd4"],
    "q017": "DELETE",
    "q018": ["h3_1ebd3300"],
    "q019": ["h2_a0de7eab", "h3_1ebd3300", "h3_58fe98b7"],
    "q020": ["h3_19c7aa03_tab", "h4_c897d9cd_tab"],
    "q021": "DELETE",
    "q024": "DELETE",
    "q025": ["h4_bd2f3d33", "h4_5d7b9ef6", "h4_a668e400_exa"],
    "q026": ["h2_feb1f085_tab", "h3_0f3c0b00_tab"],
    "q029": ["h2_88705d5b_tab", "h3_754c2a4c"],
    "q033": ["h4_09495175_tab"],
    "q036": ["h4_f6dc5631_tab"],
    "q039": ["h4_969690f3_tab"],
    "q044": ["h3_25c3ddea", "h2_5b3b7eb0"],
    "q047": "DELETE",
    "q053": ["h3_b85c5721_tab"],
    "q055": ["h3_4bce7caf_tab"],
    "q057": ["h3_6186c3fe"],
    "q058": ["h4_20778869_tab", "h4_81ba3866_tab"],
    "q173": "DELETE",
}

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    deleted_ids = []
    updated_ids = []
    kept = []

    for q in queries:
        qid = q["id"]
        if qid not in PICKS:
            kept.append(q)
            continue

        action = PICKS[qid]
        if action == "DELETE":
            deleted_ids.append(qid)
            # Don't add to kept -> query is removed
            continue

        # Update: change split, add positive_ids, remove positive_filters
        q["split"] = "train"
        q["positive_ids"] = action
        q.pop("positive_filters", None)
        q.pop("min_positive_hits", None)
        # Keep min_positives if present, or set it
        q["min_positives"] = 1
        updated_ids.append(qid)
        kept.append(q)

    # Write back
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"Deleted {len(deleted_ids)} queries: {deleted_ids}")
    print(f"Updated {len(updated_ids)} queries to train with positive_ids: {updated_ids}")

    # Count splits
    splits = {}
    for q in kept:
        s = q["split"]
        splits[s] = splits.get(s, 0) + 1
    print(f"Final split counts: {splits}")
    print(f"Total queries: {len(kept)}")

if __name__ == "__main__":
    main()
