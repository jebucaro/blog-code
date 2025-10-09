from pydantic_settings import BaseSettings, SettingsConfigDict

# Curated list of Gemini models suitable for knowledge graph extraction
AVAILABLE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


class Settings(BaseSettings):
    """Application configuration settings."""

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"

    viz_theme: str = "dark"  # Options: 'dark', 'light'

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',  # Ignore extra fields in .env
        case_sensitive=False  # Allow GEMINI_API_KEY or gemini_api_key
    )
