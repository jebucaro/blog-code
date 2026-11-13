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


def speakers_graph(rel_type, source):
    return make_graph(
        [Node(id="speaker_3", type="person"),
         Node(id="maria_lopez", type="person", label="Maria Lopez"),
         Node(id="cut_costs", type="statement"),
         Node(id="billing", type="topic")],
        [Relationship(id="r1", type=rel_type, source_node_id=source, target_node_id="cut_costs"),
         Relationship(id="r2", type="ABOUT", source_node_id="cut_costs", target_node_id="billing")],
    )


def placeholder_order_graph(speaker_first):
    a = Relationship(id="a", type="WORKS_FOR", source_node_id="speaker_2", target_node_id="acme")
    b = Relationship(id="b", type="STATED", source_node_id="speaker_2", target_node_id="cut_costs")
    return make_graph(
        [Node(id="cut_costs", type="statement"), Node(id="acme", type="organization")],
        [b, a] if speaker_first else [a, b],
    )


class TestProtectSpeakerLinks:
    def test_near_miss_speaker_is_dropped_not_remapped(self):
        repaired, report = repair_graph(speakers_graph("STATED", "speaker_2"), protect_speaker_links=True)
        assert [r.id for r in repaired.relationships] == ["r2"]
        assert report.dropped_speaker_links == 1
        assert report.remapped_endpoints == {}
        assert report.has_repairs

    def test_unknown_speaker_creates_no_placeholder(self):
        repaired, report = repair_graph(speakers_graph("PROPOSED", "unknown_speaker"), protect_speaker_links=True)
        assert "unknown_speaker" not in {n.id for n in repaired.nodes}
        assert report.placeholder_nodes == []
        assert report.dropped_speaker_links == 1

    def test_statement_survives_through_its_about_link(self):
        repaired, report = repair_graph(speakers_graph("STATED", "speaker_2"), protect_speaker_links=True)
        assert "cut_costs" in {n.id for n in repaired.nodes}
        assert "cut_costs" not in report.isolated_nodes

    def test_protected_speaker_link_resolves_by_normalized_label(self):
        repaired, report = repair_graph(speakers_graph("STATED", "Maria Lopez"), protect_speaker_links=True)
        rel = next(r for r in repaired.relationships if r.id == "r1")
        assert rel.source_node_id == "maria_lopez"
        assert report.dropped_speaker_links == 0

    def test_lowercase_speaker_link_type_is_protected(self):
        _, report = repair_graph(speakers_graph("stated", "speaker_2"), protect_speaker_links=True)
        assert report.dropped_speaker_links == 1

    def test_non_speaker_link_is_still_fuzzy_repaired(self):
        graph = make_graph(
            [Node(id="cut_costs", type="statement"), Node(id="billing_service", type="topic")],
            [Relationship(id="r1", type="ABOUT", source_node_id="cut_costs", target_node_id="billing_servic")],
        )
        repaired, report = repair_graph(graph, protect_speaker_links=True)
        assert repaired.relationships[0].target_node_id == "billing_service"
        assert report.dropped_speaker_links == 0

    def test_flag_off_keeps_todays_behaviour(self):
        repaired, report = repair_graph(speakers_graph("STATED", "speaker_2"))
        rel = next(r for r in repaired.relationships if r.id == "r1")
        assert rel.source_node_id == "speaker_3"
        assert report.dropped_speaker_links == 0

    def test_speaker_link_does_not_bind_to_placeholder_created_earlier(self):
        repaired, report = repair_graph(placeholder_order_graph(speaker_first=False), protect_speaker_links=True)
        assert [r.id for r in repaired.relationships] == ["a"]
        assert report.dropped_speaker_links == 1
        assert report.placeholder_nodes == ["speaker_2"]

    def test_speaker_link_drop_is_order_independent(self):
        first, first_report = repair_graph(placeholder_order_graph(speaker_first=False), protect_speaker_links=True)
        second, second_report = repair_graph(placeholder_order_graph(speaker_first=True), protect_speaker_links=True)
        assert {r.id for r in first.relationships} == {r.id for r in second.relationships}
        assert first_report.dropped_speaker_links == second_report.dropped_speaker_links == 1

    def test_dropped_speaker_links_alone_triggers_has_repairs(self):
        assert RepairReport(dropped_speaker_links=1).has_repairs
