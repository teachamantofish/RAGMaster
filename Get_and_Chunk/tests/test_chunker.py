import pytest


@pytest.fixture
def chunker(chunker_module):
    module = chunker_module
    module.ENABLE_CODE_EXTRACTION = True
    module.MAX_TOKENS_FOR_NODE = 10
    return module


def _linear_node(module, text, path="doc.md"):
    return ({"file_path": str(module.CWD / path)}, text)


def test_build_candidates_heading_tree(chunker):
    linear = [_linear_node(chunker, "# Title\nBody text\n")]

    candidates, front_matter = chunker.build_candidates_from_linear(linear)

    assert list(front_matter.keys()) == ["doc.md"]
    assert len(candidates) == 1
    chunk = candidates[0]
    assert chunk.heading == "Title"
    assert chunk.header_level == 1
    assert chunk.concat_header_path == "Title"
    assert chunk.parent_id is None
    assert chunk.content == "Body text"
    assert chunk.token_count == len(chunker.TOKENIZER.encode("Body text"))


def test_build_candidates_nested_and_body_nodes(chunker):
    linear = [
        _linear_node(chunker, "# Top\nIntro\n"),
        _linear_node(chunker, "## Child\nChild body\n"),
        _linear_node(chunker, "Paragraph only under child"),
    ]

    candidates, _ = chunker.build_candidates_from_linear(linear)

    ids = {c.heading: c.id for c in candidates}
    top = next(c for c in candidates if c.heading == "Top")
    child = next(c for c in candidates if c.heading == "Child")
    leaf = next(c for c in candidates if c.id not in {top.id, child.id})

    assert child.parent_id == top.id
    assert leaf.parent_id == child.id
    assert leaf.concat_header_path.endswith("Child")
    assert leaf.heading == "Child"
    assert leaf.content == "Paragraph only under child"


def test_enforce_chunk_size_peels_code_block(chunker):
    chunker.MAX_TOKENS_FOR_NODE = 5
    body = "Intro text " * 2
    code = "```python\nprint('hello')\nprint('again')\n```"
    chunk = chunker.LeafChunk(
        id="h1",
        filename="doc.md",
        parent_id=None,
        heading="Title",
        header_level=1,
        concat_header_path="Title",
        content=f"{body}\n{code}\n",
        token_count=0,
    )
    chunk.token_count = chunker._tok(chunk.content)

    final_chunks = chunker.enforce_chunk_size([chunk])

    assert len(final_chunks) == 2
    heading = final_chunks[0]
    example = final_chunks[1]

    assert heading.chunk_type == "heading"
    assert example.chunk_type == "example"
    assert example.language == "python"
    assert example.content.startswith("```python")
    assert example.id in heading.examples
    assert heading.token_count <= chunker.MAX_TOKENS_FOR_NODE


def test_enforce_chunk_size_peels_table_without_code(chunker):
    chunker.MAX_TOKENS_FOR_NODE = 5
    table = "\n".join([
        "| a | b |",
        "|---|---|",
        "| cell1 | cell2 |",
        "| cell3 | cell4 |",
    ])
    chunk = chunker.LeafChunk(
        id="h2",
        filename="doc.md",
        parent_id=None,
        heading="Has Table",
        header_level=2,
        concat_header_path="Top > Has Table",
        content=f"intro sentence before table.\n\n{table}\n",
        token_count=0,
    )
    chunk.token_count = chunker._tok(chunk.content)

    final_chunks = chunker.enforce_chunk_size([chunk])

    assert len(final_chunks) == 2
    heading = final_chunks[0]
    table_chunk = final_chunks[1]

    assert table_chunk.chunk_type == "table"
    assert table_chunk.content.startswith("| a | b |")
    assert heading.token_count <= chunker.MAX_TOKENS_FOR_NODE


def test_link_prev_next_assigns_neighbors(chunker_module):
    chunks = [
        chunker_module.LeafChunk(id="a", filename="f", parent_id=None),
        chunker_module.LeafChunk(id="b", filename="f", parent_id=None),
        chunker_module.LeafChunk(id="c", filename="f", parent_id=None),
    ]

    chunker_module.link_prev_next(chunks)

    assert chunks[0].id_prev is None and chunks[0].id_next == "b"
    assert chunks[1].id_prev == "a" and chunks[1].id_next == "c"
    assert chunks[2].id_prev == "b" and chunks[2].id_next is None


def test_chunks_to_dicts_round_trip(chunker_module):
    chunk = chunker_module.LeafChunk(
        id="n1",
        filename="doc.md",
        parent_id="p0",
        id_prev="n0",
        id_next="n2",
        heading="Heading",
        header_level=2,
        concat_header_path="Top > Heading",
        content="Body",
        examples=["ex1"],
        chunk_summary="sum",
        page_summary="page",
        language="en",
        token_count=3,
        embedding=[0.1, 0.2],
    )

    out = chunker_module.chunks_to_dicts([chunk])

    assert out == [
        {
            "id": "n1",
            "filename": "doc.md",
            "parent_id": "p0",
            "id_prev": "n0",
            "id_next": "n2",
            "heading": "Heading",
            "header_level": 2,
            "concat_header_path": "Top > Heading",
            "content": "Body",
            "examples": ["ex1"],
            "chunk_summary": "sum",
            "page_summary": "page",
            "language": "en",
            "token_count": 3,
            "embedding": [0.1, 0.2],
        }
    ]
