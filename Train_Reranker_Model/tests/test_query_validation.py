import json
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import build_retrieval_candidates as brc


def _make_chunk_map():
    return {
        "c1": {
            "id": "c1",
            "chunk_summary": "table title guidance",
            "content": "FrameMaker table title controls",
            "heading": "Tables",
        },
        "c2": {
            "id": "c2",
            "chunk_summary": "JSX SaveAsMIF script",
            "content": "Use SaveAsMIF to export",
            "heading": "Save operations",
        },
    }


def test_load_queries_enforces_min_ground_truth(tmp_path: Path):
    chunk_map = _make_chunk_map()
    queries = [
        {
            "id": "q-min",
            "query": "table title",
            "difficulty": "hard",
            "min_positives": 2,
            "positive_filters": [
                {"fields": ["chunk_summary"], "contains": "table title", "max_matches": 1}
            ],
        }
    ]
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")

    with pytest.raises(SystemExit):
        brc._load_queries(
            query_path,
            chunk_map,
            default_min_positives=1,
            default_min_positive_hits=1,
            fail_on_missing_ground_truth=True,
        )


def test_validate_positive_hits_reports_issue():
    query = brc.QueryExample(
        query_id="q-test",
        text="foo",
        positives=["c1"],
        difficulty="hard",
        split="train",
        metadata={},
        positive_filters=[],
        min_positives=1,
        min_positive_hits=1,
        positive_sources={},
    )

    issue = brc._validate_positive_hits(query, positives_in_topk=0)
    assert issue is not None
    assert issue.details["required"] == 1
    assert issue.details["found"] == 0

    no_issue = brc._validate_positive_hits(query, positives_in_topk=1)
    assert no_issue is None
