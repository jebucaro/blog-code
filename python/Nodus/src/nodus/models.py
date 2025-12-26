import logging

from pydantic import BaseModel, Field, model_validator, field_validator

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

    @field_validator('id')
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate node ID is not empty, has reasonable length, and is normalized."""
        stripped = v.strip() if v is not None else v
        if not stripped:
            raise ValueError("Node ID cannot be empty or whitespace-only")

        if len(stripped) > 200:
            raise ValueError(f"Node ID '{stripped}' is too long (max 200 characters)")

        return stripped

    @field_validator('label', 'type')
    @classmethod
    def validate_string_fields(cls, v: str | None, info) -> str | None:
        """Validate string fields have reasonable lengths and are normalized."""
        if v is None:
            return v

        field_name = info.field_name
        stripped = v.strip()

        if field_name == 'type' and not stripped:
            raise ValueError("Field 'type' cannot be empty or whitespace-only")

        if len(stripped) > 500:
            raise ValueError(f"Field '{field_name}' is too long (max 500 characters)")

        return stripped

    @model_validator(mode='after')
    def ensure_label(self):
        """Auto-generate label from id if not provided."""
        if not self.label or not self.label.strip():
            self.label = ' '.join(word.capitalize() for word in self.id.split('_'))
            logger.debug(f"Auto-generated label '{self.label}' for node id '{self.id}'")
        return self


class Relationship(BaseModel):
    """A relationship between two nodes in the knowledge graph."""
    id: str = Field(description="Unique human-readable identifier for the relationship")
    type: str = Field(description="Relationship type (e.g., 'works_at', 'located_in')")
    source_node_id: str = Field(description="ID of the source node")
    target_node_id: str = Field(description="ID of the target node")

    @field_validator('type')
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        """Validate relationship type is not empty, has reasonable length, and is normalized."""
        stripped = v.strip() if v is not None else v
        if not stripped:
            raise ValueError("Relationship type cannot be empty or whitespace-only")

        if len(stripped) > 200:
            raise ValueError(f"Relationship type '{stripped}' is too long (max 200 characters)")

        return stripped

    @field_validator('id', 'source_node_id', 'target_node_id')
    @classmethod
    def validate_ids(cls, v: str, info) -> str:
        """Validate ID fields have reasonable lengths and are normalized."""
        field_name = info.field_name.replace('_', ' ').capitalize()
        stripped = v.strip() if v is not None else v
        if not stripped:
            raise ValueError(f"{field_name} cannot be empty or whitespace-only")

        if len(stripped) > 200:
            raise ValueError(f"{field_name} '{stripped}' is too long (max 200 characters)")

        return stripped


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
        seen_semantic = set()
        unique_relationships = []
        duplicates = []

        for rel in self.relationships:
            semantic_key = (rel.source_node_id, rel.type, rel.target_node_id)

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

