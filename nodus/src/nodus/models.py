import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

NODE_TYPES: list[str] = [
    "person", "organization", "location", "event",
    "concept", "product", "technology", "date", "other",
]
RELATIONSHIP_TYPE_EXAMPLES: list[str] = [
    "WORKS_AT", "WORKS_WITH", "OWNS", "LEADS", "CREATED", "DEPENDS_ON",
    "LOCATED_IN", "PART_OF", "AFFILIATED_WITH", "ACQUIRED", "INVESTED_IN",
    "REPORTED_TO", "COMPETED_WITH", "CAUSED", "PRECEDED", "FOLLOWED", "RELATED_TO",
]

CONTEXT_MAX_LENGTH = 200


def _normalize_context(v: str | None) -> str | None:
    """Shared context-field normalization for Node and Relationship.

    Empty/whitespace-only input becomes None (so callers can rely on a single
    "no context" sentinel). Content over the length cap is soft-truncated
    with an ellipsis rather than rejected - context is supplementary, so a
    verbose LLM response shouldn't fail the whole extraction the way a bad
    id/type would.
    """
    if v is None:
        return v
    stripped = v.strip()
    if not stripped:
        return None
    if len(stripped) > CONTEXT_MAX_LENGTH:
        stripped = stripped[:CONTEXT_MAX_LENGTH].rstrip() + "…"
    return stripped


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
            "The general category of the entity, always lowercase. "
            "Must be one of: " + ", ".join(NODE_TYPES) + "."
        ),
        json_schema_extra={"enum": NODE_TYPES},
    )
    context: str | None = Field(
        default=None,
        description=(
            "A short (one sentence, under 200 characters) supporting detail from the text "
            "that isn't captured by label/type alone - e.g. a date, number, quote fragment, "
            "or key nuance. Use an empty string when there's nothing beyond the bare fact."
        )
    )

    @field_validator('id')
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate node ID is not empty, has reasonable length, and is normalized."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Node ID cannot be empty or whitespace-only")

        if len(stripped) > 200:
            raise ValueError(f"Node ID '{stripped}' is too long (max 200 characters)")

        return stripped

    @field_validator('context')
    @classmethod
    def validate_context(cls, v: str | None) -> str | None:
        """Normalize context: empty/whitespace-only becomes None, long values are truncated."""
        return _normalize_context(v)

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

        if field_name == 'type':
            normalized = stripped.lower()
            if normalized not in NODE_TYPES:
                logger.debug(f"Node type '{normalized}' not in controlled vocabulary; coercing to 'other'")
                normalized = "other"
            return normalized

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
    type: str = Field(
        description=(
            "Relationship type: a general, timeless, UPPERCASE verb phrase "
            "with underscores (e.g., 'WORKS_AT', 'LOCATED_IN')."
        )
    )
    source_node_id: str = Field(description="ID of the source node")
    target_node_id: str = Field(description="ID of the target node")
    context: str | None = Field(
        default=None,
        description=(
            "A short (one sentence, under 200 characters) supporting detail about this "
            "specific relationship - e.g. 'since 2019, as VP of Engineering' - when the text "
            "provides one. Use an empty string when there's nothing beyond the bare relationship fact."
        )
    )

    @field_validator('context')
    @classmethod
    def validate_context(cls, v: str | None) -> str | None:
        """Normalize context: empty/whitespace-only becomes None, long values are truncated."""
        return _normalize_context(v)

    @field_validator('type')
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        """Validate relationship type is not empty, has reasonable length, and is normalized."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Relationship type cannot be empty or whitespace-only")

        if len(stripped) > 200:
            raise ValueError(f"Relationship type '{stripped}' is too long (max 200 characters)")

        return stripped

    @field_validator('id', 'source_node_id', 'target_node_id')
    @classmethod
    def validate_ids(cls, v: str, info) -> str:
        """Validate ID fields have reasonable lengths and are normalized."""
        field_name = info.field_name.replace('_', ' ').title()
        stripped = v.strip()
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


def extraction_json_schema() -> dict[str, Any]:
    """KnowledgeGraph JSON schema for Gemini's structured output, with `context` required.

    The Python models keep `context` optional (repair placeholders and callers
    build nodes without it), but models tend to skip optional nullable fields
    in structured output. The schema sent to Gemini therefore requires a plain
    string per node/relationship, forcing a decision each time; an empty
    string normalizes back to None on parsing.
    """
    schema = KnowledgeGraph.model_json_schema()
    for name in ("Node", "Relationship"):
        definition = schema["$defs"][name]
        original = definition["properties"]["context"]
        definition["properties"]["context"] = {
            "type": "string",
            "title": original.get("title", "Context"),
            "description": original["description"],
        }
        definition["required"] = [*definition.get("required", []), "context"]
    return schema


def sort_relationships(relationships: list["Relationship"]) -> list["Relationship"]:
    """Return a new list of relationships grouped and alphabetized.

    Relationships are sorted by (type, source_node_id, target_node_id), each
    compared case-insensitively. The sort is stable, so relationships with an
    identical key keep their original relative order. The input list is not
    mutated. This is a presentation-only helper and has no effect on
    deduplication or storage semantics.
    """
    return sorted(
        relationships,
        key=lambda rel: (
            rel.type.lower(),
            rel.source_node_id.lower(),
            rel.target_node_id.lower(),
        ),
    )


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

