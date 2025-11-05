"""Custom exceptions for Nodus application."""


class NodusException(Exception):
    """Base exception for all Nodus errors."""

    def __init__(self, message: str, user_message: str | None = None, suggestion: str | None = None):
        """
        Initialize exception with developer and user-friendly messages.

        Args:
            message: Technical error message for logging
            user_message: User-friendly error message for display
            suggestion: Helpful suggestion for the user
        """
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message
        self.suggestion = suggestion


class APIKeyError(NodusException):
    """Raised when API key is missing or invalid."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(
            message=message,
            user_message="Invalid or missing Gemini API key",
            suggestion="Please check your API key in the sidebar settings. You can get a key from https://aistudio.google.com/apikey"
        )


class RateLimitError(NodusException):
    """Raised when API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            user_message="Too many requests to the Gemini API",
            suggestion="Please wait a moment before trying again. The free tier has rate limits."
        )


class QuotaExceededError(NodusException):
    """Raised when API quota is exceeded."""

    def __init__(self, message: str = "API quota exceeded"):
        super().__init__(
            message=message,
            user_message="Your API quota has been exceeded",
            suggestion="Please check your quota limits at https://aistudio.google.com/apikey or wait until your quota resets."
        )


class ServerOverloadedError(NodusException):
    """Raised when the API server is overloaded."""

    def __init__(self, message: str = "Server is overloaded"):
        super().__init__(
            message=message,
            user_message="The Gemini API server is currently overloaded",
            suggestion="The service is experiencing high demand. Please try again in a few moments."
        )


class NetworkError(NodusException):
    """Raised when network/connection errors occur."""

    def __init__(self, message: str = "Network error occurred"):
        super().__init__(
            message=message,
            user_message="Network connection error",
            suggestion="Please check your internet connection and try again."
        )


class ModelError(NodusException):
    """Raised when there's an issue with the model."""

    def __init__(self, message: str = "Model error"):
        super().__init__(
            message=message,
            user_message="Error with the selected AI model",
            suggestion="The model may be unavailable. Try selecting a different model from the sidebar."
        )


class ResponseParsingError(NodusException):
    """Raised when response cannot be parsed."""

    def __init__(self, message: str = "Failed to parse API response"):
        super().__init__(
            message=message,
            user_message="Failed to parse the API response",
            suggestion="The response was incomplete or malformed. Try with shorter or simpler text."
        )


class MaxTokensError(NodusException):
    """Raised when response exceeds maximum token limit."""

    def __init__(self, chars_received: int = 0):
        message = f"Response exceeded maximum token limit ({chars_received} chars received but truncated)"
        super().__init__(
            message=message,
            user_message="The knowledge graph is too large for the current model",
            suggestion="Try: (1) Using shorter input text, (2) Simplifying the content, or (3) Breaking the text into smaller chunks."
        )


class ValidationError(NodusException):
    """Raised when response validation fails."""

    def __init__(self, message: str = "Response validation failed"):
        super().__init__(
            message=message,
            user_message="The extracted data doesn't match the expected format",
            suggestion="This might be a temporary issue. Try again or use different input text."
        )


class EmptyResponseError(NodusException):
    """Raised when API returns empty response."""

    def __init__(self, message: str = "Empty response from API"):
        super().__init__(
            message=message,
            user_message="Received empty response from the API",
            suggestion="The model couldn't generate a response. Try with different or shorter text."
        )


class SafetyBlockError(NodusException):
    """Raised when content is blocked due to safety settings."""

    def __init__(self, message: str = "Content blocked by safety filters"):
        super().__init__(
            message=message,
            user_message="Content was blocked by safety filters",
            suggestion="The input text may contain content that violates safety guidelines. Try with different text."
        )


class InputTooLongError(NodusException):
    """Raised when input text is too long."""

    def __init__(self, message: str = "Input text is too long"):
        super().__init__(
            message=message,
            user_message="Input text exceeds maximum length",
            suggestion="Please reduce the length of your input text and try again."
        )
