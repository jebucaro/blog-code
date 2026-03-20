import pytest


@pytest.fixture
def sample_text() -> str:
    return (
        "Alice is a software engineer at Acme Corp. "
        "She lives in San Francisco and enjoys hiking."
    )


@pytest.fixture
def sample_node_data() -> dict:
    return {"id": "alice", "type": "person", "label": "Alice"}


@pytest.fixture
def sample_relationship_data() -> dict:
    return {
        "id": "alice_works_at_acme",
        "type": "works_at",
        "source_node_id": "alice",
        "target_node_id": "acme_corp",
    }
