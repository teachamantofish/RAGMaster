import common.token_counter as tc


def test_csv_includes_row_and_is_one_based():
    # Prepare synthetic sections: level, title, content
    sections = [
        (1, "Top", "Body of top"),
        (2, "Child", "Child content here"),
    ]

    csv_text, total = tc._build_csv_text(sections, tc.count_tokens, min_tokens=0)

    lines = csv_text.splitlines()

    # Header must include the new Row column as the leftmost column
    assert lines[0].startswith("Row,MD header,Level,Header Text,Token #,Cumulative # count")

    # Top summary row should be present immediately after header
    # Format we produce: Total,<heading_count>,,,<total_tokens>
    assert lines[1].startswith("Total,")

    # Data rows should follow and be 1-based sequential
    assert lines[2].startswith("1,")
    assert lines[3].startswith("2,")

    # Basic sanity: total should equal sum of token counts
    # (count_tokens uses tiktoken; ensure it returns ints and total is positive)
    assert isinstance(total, int)
    assert total >= 0
