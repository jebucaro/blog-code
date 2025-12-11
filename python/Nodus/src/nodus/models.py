import logging

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class Node(BaseModel):
    """Represents an entity or concept in the knowledge graph."""
    id: str = Field(
        description=(
            "The unique, standardized identifier for the entity. "
            "It MUST be lowercase and use underscores (_) as separators. For numbers like age, "
            "it MUST be prefixed (e.g., 'age_34'). MUST NOT be an integer."
        )
    )
    label: str | None = Field(
        default=None,
        description=(
            "The human-readable, capitalized name of the entity. "
            "If not provided, it will be auto-generated from the id."
        )
    )
    type: str = Field(
        description=(
            "The general type of the entity, always lowercase and singular. "
            "Example: 'person', 'occupation', 'hobby'."
        )
    )

    @model_validator(mode='after')
    def ensure_label(self):
        """Auto-generate label from id if not provided."""
        if self.label is None or self.label == '':
            # Convert id to a human-readable label: "alex_johnson" -> "Alex Johnson"
            self.label = ' '.join(word.capitalize() for word in self.id.split('_'))
            logger.debug(f"Auto-generated label '{self.label}' for node id '{self.id}'")
        return self


class Relationship(BaseModel):
    """A relationship between two nodes in the knowledge graph."""
    id: str = Field(description="Unique human-readable identifier for the relationship")
    type: str = Field(description="Relationship type (e.g., 'works_at', 'located_in')")
    source_node_id: str = Field(description="ID of the source node")
    target_node_id: str = Field(description="ID of the target node")


class KnowledgeGraph(BaseModel):
    """A knowledge graph consisting of nodes and relationships."""
    nodes: list[Node] = Field(default_factory=list, description="List of nodes in the knowledge graph")
    relationships: list[Relationship] = Field(default_factory=list,
                                              description="List of relationships in the knowledge graph")

    @model_validator(mode='after')
    def deduplicate_relationships(self):
        """Remove duplicate relationships by ID and semantic duplicates (same source, type, target)."""
        if not self.relationships:
            return self

        seen_ids = set()
        seen_semantic = set()  # Track (source_node_id, type, target_node_id) tuples
        unique_relationships = []
        duplicates = []

        for rel in self.relationships:
            semantic_key = (rel.source_node_id, rel.type, rel.target_node_id)

            # Check both ID duplicates and semantic duplicates
            if rel.id not in seen_ids and semantic_key not in seen_semantic:
                seen_ids.add(rel.id)
                seen_semantic.add(semantic_key)
                unique_relationships.append(rel)
            else:
                duplicates.append(rel)

        if duplicates:
            logger.warning(
                f"Removed {len(duplicates)} duplicate relationship(s). "
                f"Details: {[(d.id, d.source_node_id, d.type, d.target_node_id) for d in duplicates]}"
            )

        self.relationships = unique_relationships
        return self


class ExecutiveSummary(BaseModel):
    """High-level executive summary of an input document."""

    summary: str = Field(
        description=(
            "A concise, executive-level summary of the most important ideas "
            "and insights from the input text. It should be understandable "
            "on its own without reading the full document."
        )
    )
    key_points: list[str] | None = Field(
        default=None,
        description=(
            "Optional bullet-point list of the 3–7 most important points "
            "from the document, written for a busy executive."
        ),
    )


class ExtractionResult(BaseModel):
    """Bundle of summary plus knowledge graph for convenience."""

    summary: ExecutiveSummary | None = Field(
        default=None,
        description="Executive summary derived from the input text.",
    )
    knowledge_graph: KnowledgeGraph = Field(
        description="Knowledge graph extracted from either the full text or the summary.",
    )

