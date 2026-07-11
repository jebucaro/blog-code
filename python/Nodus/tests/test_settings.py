import pytest

from nodus.settings import AVAILABLE_MODELS, DEFAULT_MODEL, THINKING_LEVELS, Settings


class TestModelLineup:
    def test_available_models(self):
        assert AVAILABLE_MODELS == [
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
        ]

    def test_default_model_is_available(self):
        assert DEFAULT_MODEL == "gemini-3.5-flash"
        assert DEFAULT_MODEL in AVAILABLE_MODELS


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
