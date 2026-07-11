from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from nodus.errors import MissingAPIKeyError
from nodus.extractor import GeminiExtractor
from nodus.models import ExecutiveSummary, KnowledgeGraph
from nodus.settings import Settings


def make_extractor(**settings_overrides) -> GeminiExtractor:
    """Build a GeminiExtractor with a mocked genai client (never hits the API)."""
    settings = Settings(_env_file=None, gemini_api_key="test-key", **settings_overrides)
    with patch("nodus.extractor.genai.Client"):
        return GeminiExtractor(settings)


class TestInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        settings = Settings(_env_file=None)
        with pytest.raises(MissingAPIKeyError):
            GeminiExtractor(settings)

    def test_settings_snapshot_isolated_from_caller_mutation(self):
        settings = Settings(_env_file=None, gemini_api_key="test-key", thinking_level="default")
        with patch("nodus.extractor.genai.Client"):
            extractor = GeminiExtractor(settings)
        settings.thinking_level = "high"
        assert extractor.settings.thinking_level == "default"


class TestConfigs:
    def test_kg_config_uses_json_schema(self):
        extractor = make_extractor()
        assert extractor.kg_config.response_json_schema == KnowledgeGraph.model_json_schema()
        assert extractor.kg_config.response_mime_type == "application/json"

    def test_summary_config_uses_json_schema(self):
        extractor = make_extractor()
        assert extractor.summary_config.response_json_schema == ExecutiveSummary.model_json_schema()
        assert extractor.summary_config.response_mime_type == "application/json"

    def test_default_thinking_level_sends_no_thinking_config(self):
        extractor = make_extractor()
        assert extractor.kg_config.thinking_config is None

    @pytest.mark.parametrize(
        "level,expected",
        [
            ("low", types.ThinkingLevel.LOW),
            ("medium", types.ThinkingLevel.MEDIUM),
            ("high", types.ThinkingLevel.HIGH),
        ],
    )
    def test_explicit_thinking_level_maps_to_enum(self, level, expected):
        extractor = make_extractor(thinking_level=level)
        assert extractor.kg_config.thinking_config.thinking_level == expected

    def test_summary_config_never_gets_thinking_config(self):
        extractor = make_extractor(thinking_level="high")
        assert extractor.summary_config.thinking_config is None


class TestCacheKey:
    def test_cache_key_includes_model_and_thinking_level(self):
        extractor = make_extractor(thinking_level="high")
        key = extractor._cache_key("hello", "kg")
        assert extractor.settings.gemini_model in key
        assert "high" in key
        assert "kg" in key

    def test_different_thinking_levels_produce_different_keys(self):
        a = make_extractor(thinking_level="high")._cache_key("hello", "kg")
        b = make_extractor(thinking_level="default")._cache_key("hello", "kg")
        assert a != b


from nodus.errors import (  # noqa: E402
    APIUnavailableError,
    NetworkError,
    ParsingError,
    RateLimitError,
    TokenLimitError,
    UnknownAPIError,
)
from nodus.settings import MAX_INPUT_LENGTH  # noqa: E402


def make_response(parsed=None, text=None, finish_reason="STOP"):
    """Build a fake generate_content response."""
    response = MagicMock()
    response.parsed = parsed
    response.text = text
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    response.candidates = [candidate]
    return response


def sample_kg() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            {"id": "alice", "type": "person", "label": "Alice"},
            {"id": "acme_corp", "type": "organization", "label": "Acme Corp"},
        ],
        relationships=[
            {
                "id": "alice_works_at_acme_corp",
                "type": "WORKS_AT",
                "source_node_id": "alice",
                "target_node_id": "acme_corp",
            }
        ],
    )


class TestExtract:
    def test_returns_parsed_knowledge_graph(self):
        extractor = make_extractor()
        kg = sample_kg()
        extractor.client.models.generate_content.return_value = make_response(parsed=kg)
        result = extractor.extract("Alice works at Acme Corp.")
        assert result is kg

    def test_parsed_dict_is_validated(self):
        extractor = make_extractor()
        payload = sample_kg().model_dump()
        extractor.client.models.generate_content.return_value = make_response(parsed=payload)
        result = extractor.extract("Alice works at Acme Corp.")
        assert isinstance(result, KnowledgeGraph)
        assert result.nodes[0].id == "alice"

    def test_falls_back_to_text_when_parsed_missing(self):
        extractor = make_extractor()
        json_text = sample_kg().model_dump_json()
        extractor.client.models.generate_content.return_value = make_response(text=json_text)
        result = extractor.extract("Alice works at Acme Corp.")
        assert isinstance(result, KnowledgeGraph)
        assert len(result.nodes) == 2

    def test_caches_by_text_model_and_thinking(self):
        extractor = make_extractor()
        extractor.client.models.generate_content.return_value = make_response(parsed=sample_kg())
        extractor.extract("Alice works at Acme Corp.")
        extractor.extract("Alice works at Acme Corp.")
        assert extractor.client.models.generate_content.call_count == 1

    def test_empty_input_raises_value_error(self):
        extractor = make_extractor()
        with pytest.raises(ValueError):
            extractor.extract("   ")

    def test_too_long_input_raises_value_error(self):
        extractor = make_extractor()
        with pytest.raises(ValueError):
            extractor.extract("x" * (MAX_INPUT_LENGTH + 1))

    def test_max_tokens_raises_token_limit_error(self):
        extractor = make_extractor()
        extractor.client.models.generate_content.return_value = make_response(
            text='{"nodes": [', finish_reason="MAX_TOKENS"
        )
        with pytest.raises(TokenLimitError):
            extractor.extract("some text")

    def test_empty_response_raises_token_limit_error(self):
        extractor = make_extractor()
        extractor.client.models.generate_content.return_value = make_response()
        with pytest.raises(TokenLimitError):
            extractor.extract("some text")

    def test_invalid_json_raises_parsing_error(self):
        extractor = make_extractor()
        extractor.client.models.generate_content.return_value = make_response(text="not json")
        with pytest.raises(ParsingError):
            extractor.extract("some text")

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("503 Service Unavailable", APIUnavailableError),
            ("429 quota exceeded", RateLimitError),
            ("connection timed out", NetworkError),
            ("something exploded", UnknownAPIError),
        ],
    )
    def test_api_errors_are_mapped(self, message, expected):
        extractor = make_extractor()
        extractor.client.models.generate_content.side_effect = Exception(message)
        with pytest.raises(expected):
            extractor.extract("some text")


class TestSummarize:
    def test_returns_parsed_summary(self):
        extractor = make_extractor()
        summary = ExecutiveSummary(summary="Overview: Alice works at Acme.", key_points=["Alice"])
        extractor.client.models.generate_content.return_value = make_response(parsed=summary)
        result = extractor.summarize("Alice works at Acme Corp.")
        assert result is summary

    def test_summary_and_kg_have_separate_cache_entries(self):
        extractor = make_extractor()
        summary = ExecutiveSummary(summary="Overview: Alice works at Acme.")
        extractor.client.models.generate_content.side_effect = [
            make_response(parsed=summary),
            make_response(parsed=sample_kg()),
        ]
        extractor.summarize("Alice works at Acme Corp.")
        extractor.extract("Alice works at Acme Corp.")
        assert extractor.client.models.generate_content.call_count == 2
