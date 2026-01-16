"""Custom error types and helpers for Gemini-based extraction.

These errors are intentionally lightweight and focused on
classifying failures into user-meaningful buckets so the
UI can display clear messages while logs retain detail.
"""

from dataclasses import dataclass


@dataclass
class ExtractionError(Exception):
    """Base error for all extraction-related failures."""

    user_message: str
    detail: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.detail or self.user_message


class APIUnavailableError(ExtractionError):
    """The Gemini service is unavailable or returned a 5xx error."""


class RateLimitError(ExtractionError):
    """The Gemini API rate limit has been exceeded."""


class TokenLimitError(ExtractionError):
    """The response exceeded the token/size limits."""


class NetworkError(ExtractionError):
    """Network or connectivity issue talking to Gemini."""


class ParsingError(ExtractionError):
    """The model responded but the JSON could not be parsed/validated."""


class UnknownAPIError(ExtractionError):
    """Fallback for unexpected errors from the client/SDK."""


class MissingAPIKeyError(ExtractionError):
    """Raised when no Gemini API key is configured but required."""


def default_user_messages() -> dict[str, str]:
    """Central place for default, user-facing error messages."""

    return {
        "missing_api_key": (
            "No Gemini API key is configured. Please add your API key "
            "in the sidebar to use the extractor."
        ),
        "api_unavailable": "The Gemini service is temporarily unavailable. Please try again in a few minutes.",
        "rate_limited": "The Gemini API rate limit has been reached. Please wait a bit before trying again.",
        "token_limit": (
            "The response from Gemini was too large to process. "
            "Try using a shorter input or a more focused prompt."
        ),
        "network": "There was a network problem contacting Gemini. Check your connection and try again.",
        "parsing": (
            "Gemini returned a response, but it could not be understood. "
            "Please try simplifying the input or trying again."
        ),
        "unknown": "An unexpected error occurred while contacting Gemini.",
    }

