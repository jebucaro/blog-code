from nodus.models import KnowledgeGraph, Node, Relationship
from nodus.repair import RepairReport, repair_graph


def make_graph(nodes, rels):
    return KnowledgeGraph(nodes=nodes, relationships=rels)


class TestRepairGraph:
    def test_clean_graph_is_untouched(self):
        graph = make_graph(
            [Node(id="alice", type="person"), Node(id="acme_corp", type="organization")],
            [Relationship(id="r1", type="WORKS_AT",
                          source_node_id="alice", target_node_id="acme_corp")],
        )
        repaired, report = repair_graph(graph)
        assert repaired.model_dump() == graph.model_dump()
        assert not report.has_repairs

    def test_input_graph_is_not_mutated(self):
        graph = make_graph(
            [Node(id="alice", type="person")],
            [Relationship(id="r1", type="KNOWS",
                          source_node_id="alice", target_node_id="Betty Zhao")],
        )
        before = graph.model_dump()
        repair_graph(graph)
        assert graph.model_dump() == before

    def test_normalizable_reference_is_remapped(self):
        graph = make_graph(
            [Node(id="alice", type="person"), Node(id="acme_corp", type="organization")],
            [Relationship(id="r1", type="WORKS_AT",
                          source_node_id="alice", target_node_id="Acme Corp")],
        )
        repaired, report = repair_graph(graph)
        assert repaired.relationships[0].target_node_id == "acme_corp"
        assert report.remapped_endpoints == {"Acme Corp": "acme_corp"}
        assert report.placeholder_nodes == []

    def test_reference_matching_node_label_is_remapped(self):
        graph = make_graph(
            [Node(id="alice", type="person"),
             Node(id="acme_corp", type="organization", label="Acme Corporation")],
            [Relationship(id="r1", type="WORKS_AT",
                          source_node_id="alice", target_node_id="acme_corporation")],
        )
        repaired, report = repair_graph(graph)
        assert repaired.relationships[0].target_node_id == "acme_corp"
        assert report.remapped_endpoints == {"acme_corporation": "acme_corp"}

    def test_fuzzy_reference_is_remapped(self):
        graph = make_graph(
            [Node(id="john_smith", type="person"), Node(id="mary", type="person")],
            [Relationship(id="r1", type="WORKS_WITH",
                          source_node_id="john_smyth", target_node_id="mary")],
        )
        repaired, report = repair_graph(graph)
        assert repaired.relationships[0].source_node_id == "john_smith"
        assert report.remapped_endpoints == {"john_smyth": "john_smith"}

    def test_unmatched_reference_creates_placeholder(self):
        graph = make_graph(
            [Node(id="alice", type="person")],
            [Relationship(id="r1", type="KNOWS",
                          source_node_id="alice", target_node_id="Betty Zhao")],
        )
        repaired, report = repair_graph(graph)
        assert report.placeholder_nodes == ["betty_zhao"]
        placeholder = next(n for n in repaired.nodes if n.id == "betty_zhao")
        assert placeholder.type == "other"
        assert placeholder.label == "Betty Zhao"
        assert repaired.relationships[0].target_node_id == "betty_zhao"

    def test_placeholder_is_reused_across_edges(self):
        graph = make_graph(
            [Node(id="alice", type="person"), Node(id="bob", type="person")],
            [
                Relationship(id="r1", type="KNOWS",
                             source_node_id="alice", target_node_id="Betty Zhao"),
                Relationship(id="r2", type="KNOWS",
                             source_node_id="bob", target_node_id="betty_zhao"),
            ],
        )
        repaired, report = repair_graph(graph)
        assert report.placeholder_nodes == ["betty_zhao"]
        assert len([n for n in repaired.nodes if n.id == "betty_zhao"]) == 1

    def test_self_loop_is_dropped_and_counted(self):
        graph = make_graph(
            [Node(id="alice", type="person"), Node(id="acme", type="organization")],
            [
                Relationship(id="r1", type="KNOWS",
                             source_node_id="alice", target_node_id="alice"),
                Relationship(id="r2", type="WORKS_AT",
                             source_node_id="alice", target_node_id="acme"),
            ],
        )
        repaired, report = repair_graph(graph)
        assert report.dropped_self_loops == 1
        assert len(repaired.relationships) == 1
        assert repaired.relationships[0].id == "r2"

    def test_isolated_nodes_are_reported_not_removed(self):
        graph = make_graph(
            [Node(id="alice", type="person"), Node(id="acme", type="organization"),
             Node(id="loner", type="concept")],
            [Relationship(id="r1", type="WORKS_AT",
                          source_node_id="alice", target_node_id="acme")],
        )
        repaired, report = repair_graph(graph)
        assert report.isolated_nodes == ["loner"]
        assert {n.id for n in repaired.nodes} == {"alice", "acme", "loner"}


class TestRepairReport:
    def test_empty_report_has_no_repairs(self):
        assert not RepairReport().has_repairs

    def test_any_field_triggers_has_repairs(self):
        assert RepairReport(dropped_self_loops=1).has_repairs
        assert RepairReport(placeholder_nodes=["x"]).has_repairs
        assert RepairReport(remapped_endpoints={"a": "b"}).has_repairs
        assert RepairReport(isolated_nodes=["x"]).has_repairs
