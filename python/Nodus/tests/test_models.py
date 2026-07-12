import pytest
from pydantic import ValidationError

from nodus.models import KnowledgeGraph, Node, Relationship, NODE_TYPES


class TestNode:
    def test_valid_node(self):
        node = Node(id="alice", type="person", label="Alice")
        assert node.id == "alice"
        assert node.type == "person"
        assert node.label == "Alice"

    def test_label_auto_generated_from_id(self):
        node = Node(id="acme_corp", type="organization")
        assert node.label == "Acme Corp"

    def test_label_auto_generated_multi_word(self):
        node = Node(id="san_francisco_bay", type="location")
        assert node.label == "San Francisco Bay"

    def test_id_whitespace_stripped(self):
        node = Node(id="  alice  ", type="person")
        assert node.id == "alice"

    def test_type_whitespace_stripped(self):
        node = Node(id="alice", type="  person  ")
        assert node.type == "person"

    def test_label_whitespace_stripped(self):
        node = Node(id="alice", type="person", label="  Alice  ")
        assert node.label == "Alice"

    def test_empty_id_raises(self):
        with pytest.raises(ValidationError, match="Node ID cannot be empty"):
            Node(id="", type="person")

    def test_whitespace_only_id_raises(self):
        with pytest.raises(ValidationError, match="Node ID cannot be empty"):
            Node(id="   ", type="person")

    def test_id_too_long_raises(self):
        with pytest.raises(ValidationError, match="too long"):
            Node(id="a" * 201, type="person")

    def test_empty_type_raises(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Node(id="alice", type="")

    def test_type_too_long_raises(self):
        with pytest.raises(ValidationError, match="too long"):
            Node(id="alice", type="t" * 501)

    def test_label_too_long_raises(self):
        with pytest.raises(ValidationError, match="too long"):
            Node(id="alice", type="person", label="l" * 501)

    def test_label_none_triggers_auto_generation(self):
        node = Node(id="hello_world", type="concept", label=None)
        assert node.label == "Hello World"

    def test_label_empty_string_triggers_auto_generation(self):
        node = Node(id="hello_world", type="concept", label="")
        assert node.label == "Hello World"


class TestRelationship:
    def test_valid_relationship(self):
        rel = Relationship(
            id="alice_works_at_acme",
            type="works_at",
            source_node_id="alice",
            target_node_id="acme_corp",
        )
        assert rel.id == "alice_works_at_acme"
        assert rel.type == "works_at"
        assert rel.source_node_id == "alice"
        assert rel.target_node_id == "acme_corp"

    def test_empty_type_raises(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Relationship(id="r1", type="", source_node_id="a", target_node_id="b")

    def test_type_too_long_raises(self):
        with pytest.raises(ValidationError, match="too long"):
            Relationship(id="r1", type="t" * 201, source_node_id="a", target_node_id="b")

    def test_empty_id_raises(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Relationship(id="", type="works_at", source_node_id="a", target_node_id="b")

    def test_empty_source_node_id_raises(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Relationship(id="r1", type="works_at", source_node_id="", target_node_id="b")

    def test_empty_target_node_id_raises(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Relationship(id="r1", type="works_at", source_node_id="a", target_node_id="")

    def test_ids_whitespace_stripped(self):
        rel = Relationship(
            id="  r1  ",
            type="works_at",
            source_node_id="  alice  ",
            target_node_id="  acme  ",
        )
        assert rel.id == "r1"
        assert rel.source_node_id == "alice"
        assert rel.target_node_id == "acme"


class TestKnowledgeGraph:
    def test_empty_graph(self):
        kg = KnowledgeGraph()
        assert kg.nodes == []
        assert kg.relationships == []

    def test_graph_with_nodes_and_relationships(self):
        nodes = [
            Node(id="alice", type="person"),
            Node(id="acme_corp", type="organization"),
        ]
        rels = [
            Relationship(
                id="r1", type="works_at", source_node_id="alice", target_node_id="acme_corp"
            )
        ]
        kg = KnowledgeGraph(nodes=nodes, relationships=rels)
        assert len(kg.nodes) == 2
        assert len(kg.relationships) == 1

    def test_duplicate_relationships_by_id_removed(self):
        rels = [
            Relationship(id="r1", type="works_at", source_node_id="alice", target_node_id="acme"),
            Relationship(id="r1", type="works_at", source_node_id="alice", target_node_id="acme"),
        ]
        kg = KnowledgeGraph(relationships=rels)
        assert len(kg.relationships) == 1

    def test_semantic_duplicate_relationships_removed(self):
        rels = [
            Relationship(id="r1", type="works_at", source_node_id="alice", target_node_id="acme"),
            Relationship(id="r2", type="works_at", source_node_id="alice", target_node_id="acme"),
        ]
        kg = KnowledgeGraph(relationships=rels)
        assert len(kg.relationships) == 1
        assert kg.relationships[0].id == "r1"

    def test_non_duplicate_relationships_kept(self):
        rels = [
            Relationship(id="r1", type="works_at", source_node_id="alice", target_node_id="acme"),
            Relationship(id="r2", type="lives_in", source_node_id="alice", target_node_id="sf"),
        ]
        kg = KnowledgeGraph(relationships=rels)
        assert len(kg.relationships) == 2


class TestKnowledgeGraphSchema:
    def test_node_type_enum_in_schema(self):
        schema = KnowledgeGraph.model_json_schema()
        node_type_schema = schema["$defs"]["Node"]["properties"]["type"]
        assert node_type_schema["enum"] == NODE_TYPES

    def test_node_type_description_uses_controlled_vocabulary(self):
        schema = KnowledgeGraph.model_json_schema()
        desc = schema["$defs"]["Node"]["properties"]["type"]["description"]
        assert "person" in desc
        assert "occupation" not in desc and "hobby" not in desc

    def test_relationship_type_description_uses_uppercase_examples(self):
        schema = KnowledgeGraph.model_json_schema()
        desc = schema["$defs"]["Relationship"]["properties"]["type"]["description"]
        assert "WORKS_AT" in desc
        assert "'works_at'" not in desc

    def test_unknown_node_type_still_coerces_to_other(self):
        node = Node(id="alice", type="astronaut_dog")
        assert node.type == "other"
