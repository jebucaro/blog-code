import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from models import KnowledgeGraph, ExecutiveSummary, ExtractionResult
from settings import Settings
from errors import (
    APIUnavailableError,
    ExtractionError,
    MissingAPIKeyError,
    NetworkError,
    ParsingError,
    RateLimitError,
    TokenLimitError,
    UnknownAPIError,
    default_user_messages,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
# Knowledge Graph Extraction Expert for Gemini 2.5+

## 1. Your Role
You are an expert system for extracting structured information to build a knowledge graph. Your goal is to capture all meaningful entities and relationships from the input text with high accuracy, adhering strictly to the provided JSON schema.

## 2. Input Text
The input text may be a raw document or a pre-processed, structured summary with headings like "Key Entities" and "Key Relationships." Use these structural hints to your advantage, but always extract information from the text's content, not the headings themselves.

## 3. Core Principles
- **Completeness:** Extract all distinct entities and the relationships connecting them. Do not add any information that is not in the text.
- **Accuracy:** Ensure every relationship's `source_node_id` and `target_node_id` correctly references an `id` from the `nodes` list. If an entity in a relationship does not exist as a node, you must create it.
- **Consistency:** Use the same `id` for an entity across all nodes and relationships. For example, if "Dr. Alex Johnson" is also called "Alex," both should resolve to the same node with the `id` 'alex_johnson'.

## 4. Node Generation Rules
- **`id` (Standardized Key):**
    - Generate the `id` by converting the entity's name to **lowercase** and replacing all spaces and special characters with **underscores (_)**.
    - For numeric concepts (e.g., "34 years old"), the `id` must be prefixed (e.g., 'age_34').
    - Never use a standalone integer as an `id`.
- **`label` (Human-Readable Name):**
    - Use the original, human-readable name of the entity as the `label`. For example, if the text says "Dr. Alex Johnson," the `label` should be "Dr. Alex Johnson".
- **`type` (General Category):**
    - The `type` must be a basic, **lowercase**, singular category (e.g., 'person', 'organization', 'date'). Avoid overly specific types like 'mathematician'.

## 5. Relationship Generation Rules
- **`id` (Unique Identifier):**
    - Create a unique, human-readable identifier for each relationship (e.g., 'acme_corp_works_with_vendor_x').
- **`type` (Relationship Type):**
    - The `type` must be a general, timeless, and **UPPERCASE** verb phrase using **underscores (_)** (e.g., 'WORKS_AS', 'DEPENDS_ON').
"""


SUMMARY_SYSTEM_PROMPT = """
You are an expert executive assistant creating a structured briefing document.

Goal:
- Read the input text and produce a concise, fact-based briefing document
  that is optimized for both human scanning and downstream machine parsing.
- You MUST NOT create or output a knowledge graph, only structured text.

Audience:
- A busy executive who needs the key facts, decisions, and risks.
- An automated system that will parse this document to build a knowledge graph.

General requirements:
- Be concise but information-dense.
- Preserve all key proper nouns (people, organizations, projects), numbers, and dates.
- Use clear, literal language. Avoid metaphors or speculation.

Text structure (very important):
- Organize the `summary` text into the following labeled sections,
  in this exact order, using these headings:

  1. "Overview:"
  2. "Key Entities and Roles:"
  3. "Decisions and Actions:"
  4. "Key Relationships:"
  5. "Risks and Constraints:"

- Each section should contain 1-3 concise sentences.
- When possible, write sentences in an active voice with a simple
  "Subject-Verb-Object" structure (e.g., "Company A acquired Company B.").

Style constraints:
- Minimize the use of pronouns ("it", "they"). Repeat entity names for clarity.
- Prefer simple, direct verbs (e.g., "leads," "owns," "depends on," "affects").
- Avoid low-level technical details unless essential for understanding a key point.

JSON output format:
- Return a JSON object matching the provided schema.
- The `summary` field must contain the full structured text with the
  five headings above.
- Optionally include `key_points` (3-7 bullet-style strings) that
  highlight the most critical facts for an executive.
"""


class GeminiExtractor:
    """Extract knowledge graphs from text using Google Gemini API."""

    def __init__(self, settings: Settings | None = None, api_key: str | None = None):
        """
        Initialize the GeminiExtractor

        Args:
            settings (Settings | None): Configuration object containing API key and model name.
            api_key (str | None): Direct API key string. If provided, it overrides config.
        """
        self.settings = settings or Settings()

        key_to_use = api_key or self.settings.gemini_api_key
        messages = default_user_messages()
        if not key_to_use:
            # Explicit, user-friendly error for missing key
            raise MissingAPIKeyError(
                user_message=messages["missing_api_key"],
                detail="Gemini API key must be provided via argument or Settings.gemini_api_key.",
            )

        self.client = genai.Client(api_key=key_to_use)
        self.kg_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=KnowledgeGraph.model_json_schema(),
        )
        self.summary_config = types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ExecutiveSummary.model_json_schema(),
        )

        logger.info("Initialized Gemini extractor")

    def extract(self, text: str) -> KnowledgeGraph:
        """
        Extract a knowledge graph from the provided text.

        Args:
            text (str): The input text to extract the knowledge graph from.

        Returns:
            KnowledgeGraph: The extracted knowledge graph.
        """

        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # NOTE: API-key presence is not enforced here to
        # allow free-tier / key-less usage when supported
        # by the underlying Google Gemini service.

        json_data: str | None = None
        messages = default_user_messages()

        try:
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=text,
                config=self.kg_config,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Gemini API responded in {elapsed_time:.2f}s")

            # Check if response was truncated or incomplete
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                finish_reason: Any = getattr(candidate, "finish_reason", None)

                if finish_reason and finish_reason != "STOP":
                    logger.warning(f"Response may be incomplete. Finish reason: {finish_reason}")

            # Get text from response
            try:
                json_data = response.text
            except Exception as e:  # Fallback for older/changed SDKs
                logger.error(f"Error accessing response.text: {e}")
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None) if content is not None else None
                    if parts and hasattr(parts[0], "text"):
                        json_data = parts[0].text

            if not json_data:
                # Treat empty responses as token/size or service issues
                raise TokenLimitError(
                    user_message=messages["token_limit"],
                    detail="Empty response from Gemini API; likely token or size limit.",
                )

            # Check for MAX_TOKENS after retrieving text
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == "MAX_TOKENS":
                    logger.error(f"Response truncated at {len(json_data)} characters")
                    logger.error(f"Last 200 chars: {json_data[-200:]}")
                    raise TokenLimitError(
                        user_message=messages["token_limit"],
                        detail=(
                            f"Response exceeded maximum token limit (finish_reason=MAX_TOKENS, chars={len(json_data)})."
                        ),
                    )

            parsed_data = json.loads(json_data)

            # Log raw response for debugging (minified)
            logger.debug(f"Raw Gemini response: {json.dumps(parsed_data, separators=(',', ':'))}")

            knowledge_graph = KnowledgeGraph.model_validate(parsed_data)

            logger.info(
                "Successfully extracted knowledge graph with %d nodes and %d relationships",
                len(knowledge_graph.nodes),
                len(knowledge_graph.relationships),
            )
            return knowledge_graph

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON response from Gemini: %s", e)
            if json_data is not None:
                logger.error("Response length: %d characters", len(json_data))
                logger.error("Last 500 chars of response: %s", json_data[-500:])
            raise ParsingError(
                user_message=messages["parsing"],
                detail=str(e),
            ) from e

        except ParsingError:
            # Already mapped; just propagate.
            raise

        except TokenLimitError:
            # Already mapped; just propagate.
            raise

        except Exception as e:
            # Map common SDK/network-ish errors to user-friendly classes.
            detail = str(e)
            lowered = detail.lower()

            # Very lightweight heuristic mapping without depending on
            # internal Google SDK exception types.
            if any(code in lowered for code in ("unavailable", "503", "502")):
                mapped: ExtractionError = APIUnavailableError(
                    user_message=messages["api_unavailable"],
                    detail=detail,
                )
            elif any(term in lowered for term in ("rate limit", "quota", "429")):
                mapped = RateLimitError(
                    user_message=messages["rate_limited"],
                    detail=detail,
                )
            elif any(term in lowered for term in ("timeout", "timed out", "connection", "network")):
                mapped = NetworkError(
                    user_message=messages["network"],
                    detail=detail,
                )
            else:
                mapped = UnknownAPIError(
                    user_message=messages["unknown"],
                    detail=detail,
                )

            logger.error("Extraction failed: %s", mapped)
            raise mapped from e

    def summarize(self, text: str) -> ExecutiveSummary:
        """Create an executive summary from the provided text using Gemini.

        The result is structured according to the ExecutiveSummary model.
        """

        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        json_data: str | None = None
        messages = default_user_messages()

        try:
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=text,
                config=self.summary_config,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Gemini summary responded in {elapsed_time:.2f}s")

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                finish_reason: Any = getattr(candidate, "finish_reason", None)
                if finish_reason and finish_reason != "STOP":
                    logger.warning(f"Summary response may be incomplete. Finish reason: {finish_reason}")

            try:
                json_data = response.text
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Error accessing summary response.text: {e}")
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None) if content is not None else None
                    if parts and hasattr(parts[0], "text"):
                        json_data = parts[0].text

            if not json_data:
                raise TokenLimitError(
                    user_message=messages["token_limit"],
                    detail="Empty response from Gemini summary API; likely token or size limit.",
                )

            if hasattr(response, "candidates") and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == "MAX_TOKENS":
                    logger.error(f"Summary response truncated at {len(json_data)} characters")
                    logger.error(f"Summary last 200 chars: {json_data[-200:]}")
                    raise TokenLimitError(
                        user_message=messages["token_limit"],
                        detail=(
                            "Summary response exceeded maximum token limit "
                            f"(finish_reason=MAX_TOKENS, chars={len(json_data)})."
                        ),
                    )

            parsed_data = json.loads(json_data)
            logger.debug(f"Raw Gemini summary response: {json.dumps(parsed_data, separators=(',', ':'))}")

            summary = ExecutiveSummary.model_validate(parsed_data)
            logger.info("Successfully generated executive summary")
            return summary

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON summary response from Gemini: %s", e)
            if json_data is not None:
                logger.error("Summary response length: %d characters", len(json_data))
                logger.error("Summary last 500 chars of response: %s", json_data[-500:])
            raise ParsingError(
                user_message=messages["parsing"],
                detail=str(e),
            ) from e

        except ParsingError:
            raise

        except TokenLimitError:
            raise

        except Exception as e:  # pragma: no cover - error mapping
            detail = str(e)
            lowered = detail.lower()

            if any(code in lowered for code in ("unavailable", "503", "502")):
                mapped: ExtractionError = APIUnavailableError(
                    user_message=messages["api_unavailable"],
                    detail=detail,
                )
            elif any(term in lowered for term in ("rate limit", "quota", "429")):
                mapped = RateLimitError(
                    user_message=messages["rate_limited"],
                    detail=detail,
                )
            elif any(term in lowered for term in ("timeout", "timed out", "connection", "network")):
                mapped = NetworkError(
                    user_message=messages["network"],
                    detail=detail,
                )
            else:
                mapped = UnknownAPIError(
                    user_message=messages["unknown"],
                    detail=detail,
                )

            logger.error("Summary generation failed: %s", mapped)
            raise mapped from e

    def extract_with_summary(self, text: str, use_summary_for_kg: bool = True) -> ExtractionResult:
        """High-level helper that performs summarization and KG extraction.

        Args:
            text: Original input text.
            use_summary_for_kg: If True, build the KG from the summary
                text; otherwise, build it directly from the full text.
        """

        summary: ExecutiveSummary | None = None

        # Always produce a summary so the UI can show it.
        summary = self.summarize(text)

        if use_summary_for_kg:
            kg_input = summary.summary
        else:
            kg_input = text

        knowledge_graph = self.extract(kg_input)
        return ExtractionResult(summary=summary, knowledge_graph=knowledge_graph)

    def close(self):
        """Properly close the Gemini client to avoid cleanup errors."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def __del__(self):
        """Destructor to ensure the client is closed."""
        self.close()
