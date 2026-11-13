import pytest

from nodus.settings import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    INPUT_TYPES,
    THINKING_LEVELS,
    Settings,
)


class TestModelLineup:
    def test_available_models(self):
        assert AVAILABLE_MODELS == [
            "gemini-3.5-flash-lite",
            "gemini-4-hydrogen",
            "gemini-4-carbon",
        ]

    def test_default_model_is_available(self):
        assert DEFAULT_MODEL == "gemini-4-hydrogen"
        assert DEFAULT_MODEL in AVAILABLE_MODELS

    @pytest.mark.parametrize("removed", ["gemini-3.8-flash", "gemini-4-argon"])
    def test_removed_models_not_offered(self, removed):
        # Removed IDs must stay out of the picker; app.py falls back to DEFAULT_MODEL for them.
        assert removed not in AVAILABLE_MODELS

    def test_thinking_levels_valid_on_every_model(self):
        # "minimal" is rejected by Gemini 4 models; "xhigh" only exists on Hydrogen.
        assert "minimal" not in THINKING_LEVELS
        assert "xhigh" not in THINKING_LEVELS


class TestSettings:
    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        settings = Settings(_env_file=None)
        assert settings.gemini_model == DEFAULT_MODEL

    def test_default_thinking_level(self, monkeypatch):
        monkeypatch.delenv("THINKING_LEVEL", raising=False)
        settings = Settings(_env_file=None)
        assert settings.thinking_level == "default"

    @pytest.mark.parametrize("level", ["default", "low", "medium", "high"])
    def test_valid_thinking_levels_accepted(self, level):
        settings = Settings(_env_file=None, thinking_level=level)
        assert settings.thinking_level == level

    def test_thinking_level_normalized(self):
        settings = Settings(_env_file=None, thinking_level="  HIGH ")
        assert settings.thinking_level == "high"

    def test_invalid_thinking_level_rejected(self):
        with pytest.raises(ValueError):
            Settings(_env_file=None, thinking_level="ultra")

    def test_use_thinking_field_removed(self, monkeypatch):
        monkeypatch.delenv("USE_THINKING", raising=False)
        settings = Settings(_env_file=None)
        assert "use_thinking" not in type(settings).model_fields

    def test_thinking_levels_constant(self):
        assert THINKING_LEVELS == ["default", "low", "medium", "high"]


class TestInputType:
    def test_input_types_constant(self):
        assert INPUT_TYPES == ["document", "transcript"]

    def test_default_is_document(self, monkeypatch):
        monkeypatch.delenv("INPUT_TYPE", raising=False)
        assert Settings(_env_file=None).input_type == "document"

    @pytest.mark.parametrize("value", ["document", "transcript"])
    def test_valid_values_accepted(self, value):
        assert Settings(_env_file=None, input_type=value).input_type == value

    def test_value_is_normalized(self):
        assert Settings(_env_file=None, input_type=" Transcript ").input_type == "transcript"

    def test_env_var_is_read(self, monkeypatch):
        monkeypatch.setenv("INPUT_TYPE", "TRANSCRIPT")
        assert Settings(_env_file=None).input_type == "transcript"

    def test_invalid_value_rejected_with_allowed_values(self):
        with pytest.raises(ValueError, match="document.*transcript"):
            Settings(_env_file=None, input_type="meeting")
