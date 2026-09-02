from unittest.mock import MagicMock, patch

import pytest

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
        response_format = extractor.kg_config["response_format"]
        assert response_format["schema"] == KnowledgeGraph.model_json_schema()
        assert response_format["mime_type"] == "application/json"

    def test_summary_config_uses_json_schema(self):
        extractor = make_extractor()
        response_format = extractor.summary_config["response_format"]
        assert response_format["schema"] == ExecutiveSummary.model_json_schema()
        assert response_format["mime_type"] == "application/json"

    def test_default_thinking_level_sends_no_generation_config(self):
        extractor = make_extractor()
        assert "generation_config" not in extractor.kg_config

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_explicit_thinking_level_sets_generation_config(self, level):
        extractor = make_extractor(thinking_level=level)
        assert extractor.kg_config["generation_config"] == {"thinking_level": level}

    def test_summary_config_never_gets_generation_config(self):
        extractor = make_extractor(thinking_level="high")
        assert "generation_config" not in extractor.summary_config

    def test_configs_do_not_send_safety_settings(self):
        """The standard Gemini API rejects safety_settings on the Interactions
        endpoint (400 invalid_request) - only Gemini Enterprise supports it."""
        extractor = make_extractor()
        assert "safety_settings" not in extractor.kg_config
        assert "safety_settings" not in extractor.summary_config


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


def make_response(output_text=None, status="completed"):
    """Build a fake Interactions API response."""
    response = MagicMock()
    response.output_text = output_text
    response.status = status
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
        extractor.client.interactions.create.return_value = make_response(output_text=kg.model_dump_json())
        result = extractor.extract("Alice works at Acme Corp.")
        assert isinstance(result, KnowledgeGraph)
        assert result.nodes[0].id == "alice"

    def test_caches_by_text_model_and_thinking(self):
        extractor = make_extractor()
        extractor.client.interactions.create.return_value = make_response(
            output_text=sample_kg().model_dump_json()
        )
        extractor.extract("Alice works at Acme Corp.")
        extractor.extract("Alice works at Acme Corp.")
        assert extractor.client.interactions.create.call_count == 1

    def test_empty_input_raises_value_error(self):
        extractor = make_extractor()
        with pytest.raises(ValueError):
            extractor.extract("   ")

    def test_too_long_input_raises_value_error(self):
        extractor = make_extractor()
        with pytest.raises(ValueError):
            extractor.extract("x" * (MAX_INPUT_LENGTH + 1))

    def test_incomplete_status_raises_token_limit_error(self):
        extractor = make_extractor()
        extractor.client.interactions.create.return_value = make_response(
            output_text='{"nodes": [', status="incomplete"
        )
        with pytest.raises(TokenLimitError):
            extractor.extract("some text")

    def test_empty_response_raises_token_limit_error(self):
        extractor = make_extractor()
        extractor.client.interactions.create.return_value = make_response()
        with pytest.raises(TokenLimitError):
            extractor.extract("some text")

    def test_invalid_json_raises_parsing_error(self):
        extractor = make_extractor()
        extractor.client.interactions.create.return_value = make_response(output_text="not json")
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
        extractor.client.interactions.create.side_effect = Exception(message)
        with pytest.raises(expected):
            extractor.extract("some text")


class TestSummarize:
    def test_returns_parsed_summary(self):
        extractor = make_extractor()
        summary = ExecutiveSummary(summary="Overview: Alice works at Acme.", key_points=["Alice"])
        extractor.client.interactions.create.return_value = make_response(
            output_text=summary.model_dump_json()
        )
        result = extractor.summarize("Alice works at Acme Corp.")
        assert result == summary

    def test_summary_and_kg_have_separate_cache_entries(self):
        extractor = make_extractor()
        summary = ExecutiveSummary(summary="Overview: Alice works at Acme.")
        extractor.client.interactions.create.side_effect = [
            make_response(output_text=summary.model_dump_json()),
            make_response(output_text=sample_kg().model_dump_json()),
        ]
        extractor.summarize("Alice works at Acme Corp.")
        extractor.extract("Alice works at Acme Corp.")
        assert extractor.client.interactions.create.call_count == 2
