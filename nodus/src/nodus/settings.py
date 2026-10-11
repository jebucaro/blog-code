from dotenv import find_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Curated list of Gemini models suitable for knowledge graph extraction,
# ordered fastest -> most capable
AVAILABLE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
]

DEFAULT_MODEL = "gemini-3.8-flash"

# Thinking levels the app exposes. "default" sends no thinking config so each
# model uses its built-in default (MEDIUM on flash, HIGH on pro, MINIMAL on
# flash-lite). MINIMAL is never offered: gemini-3.1-pro does not support it.
THINKING_LEVELS = ["default", "low", "medium", "high"]

# Maximum input text length (characters) for extraction/summarization
# Approximately 1 hour of transcription at typical speaking rate
MAX_INPUT_LENGTH = 100000


class Settings(BaseSettings):
    """Application configuration settings."""

    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_MODEL
    thinking_level: str = "default"

    viz_theme: str = "dark"  # Options: 'dark', 'light'

    model_config = SettingsConfigDict(
        env_file=find_dotenv(usecwd=True) or '.env',
        env_file_encoding='utf-8',
        extra='ignore',  # Ignore extra fields in .env
        case_sensitive=False  # Allow GEMINI_API_KEY or gemini_api_key
    )

    @field_validator('thinking_level')
    @classmethod
    def validate_thinking_level(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in THINKING_LEVELS:
            raise ValueError(
                f"thinking_level must be one of {THINKING_LEVELS}, got '{v}'"
            )
        return normalized
