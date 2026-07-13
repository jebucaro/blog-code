import difflib
import logging
import re

from pydantic import BaseModel, Field

from nodus.models import KnowledgeGraph, Node, Relationship

logger = logging.getLogger(__name__)

# High cutoff so fuzzy matching fixes typos without merging distinct entities.
FUZZY_MATCH_CUTOFF = 0.85


class RepairReport(BaseModel):
    """Summary of repairs applied to a knowledge graph before display."""

    remapped_endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dangling endpoint reference -> existing node id it was remapped to.",
    )
    placeholder_nodes: list[str] = Field(
        default_factory=list,
        description="IDs of placeholder nodes created for unmatched endpoints.",
    )
    dropped_self_loops: int = 0
    isolated_nodes: list[str] = Field(
        default_factory=list,
        description="IDs of nodes with no relationships after repair.",
    )

    @property
    def has_repairs(self) -> bool:
        return bool(
            self.remapped_endpoints
            or self.placeholder_nodes
            or self.dropped_self_loops
            or self.isolated_nodes
        )


def _normalize_id(ref: str) -> str:
    """Apply the prompt's id rules: lowercase, non-alphanumerics collapsed to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", ref.strip().lower()).strip("_")


def repair_graph(graph: KnowledgeGraph) -> tuple[KnowledgeGraph, RepairReport]:
    """Fix dangling relationship endpoints instead of letting them be silently dropped.

    Resolution order per endpoint: exact node id -> normalized id/label match ->
    fuzzy match (difflib, cutoff FUZZY_MATCH_CUTOFF) -> new placeholder node
    (type 'other'). Self-loops are dropped and counted; isolated nodes are kept
    and reported. The input graph is never mutated.
    """
    report = RepairReport()

    nodes = [node.model_copy() for node in graph.nodes]
    node_ids = {node.id for node in nodes}

    candidates: dict[str, str] = {}
    for node in nodes:
        candidates.setdefault(_normalize_id(node.id), node.id)
        if node.label:
            candidates.setdefault(_normalize_id(node.label), node.id)

    def resolve(ref: str) -> str:
        if ref in node_ids:
            return ref
        normalized = _normalize_id(ref)
        if normalized in candidates:
            report.remapped_endpoints[ref] = candidates[normalized]
            return candidates[normalized]
        close = difflib.get_close_matches(
            normalized, list(candidates), n=1, cutoff=FUZZY_MATCH_CUTOFF
        )
        if close:
            report.remapped_endpoints[ref] = candidates[close[0]]
            return candidates[close[0]]
        placeholder_id = normalized or ref
        if placeholder_id not in node_ids:
            nodes.append(Node(id=placeholder_id, type="other"))
            node_ids.add(placeholder_id)
            candidates.setdefault(placeholder_id, placeholder_id)
            report.placeholder_nodes.append(placeholder_id)
        if ref != placeholder_id:
            report.remapped_endpoints[ref] = placeholder_id
        return placeholder_id

    repaired_rels: list[Relationship] = []
    for rel in graph.relationships:
        source = resolve(rel.source_node_id)
        target = resolve(rel.target_node_id)
        if source == target:
            report.dropped_self_loops += 1
            continue
        repaired_rels.append(
            rel.model_copy(update={"source_node_id": source, "target_node_id": target})
        )

    connected = {r.source_node_id for r in repaired_rels} | {
        r.target_node_id for r in repaired_rels
    }
    report.isolated_nodes = [node.id for node in nodes if node.id not in connected]

    repaired = KnowledgeGraph(nodes=nodes, relationships=repaired_rels)
    if report.has_repairs:
        logger.info(
            "Graph repair: %d endpoint(s) remapped, %d placeholder(s) created, "
            "%d self-loop(s) dropped, %d isolated node(s)",
            len(report.remapped_endpoints),
            len(report.placeholder_nodes),
            report.dropped_self_loops,
            len(report.isolated_nodes),
        )
    return repaired, report
