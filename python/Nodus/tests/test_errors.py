import pytest

from nodus.errors import (
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


class TestExtractionError:
    def test_is_exception(self):
        err = ExtractionError(user_message="Something went wrong")
        assert isinstance(err, Exception)

    def test_user_message_stored(self):
        err = ExtractionError(user_message="Something went wrong")
        assert err.user_message == "Something went wrong"

    def test_detail_defaults_to_none(self):
        err = ExtractionError(user_message="Something went wrong")
        assert err.detail is None

    def test_detail_stored_when_provided(self):
        err = ExtractionError(user_message="Something went wrong", detail="Internal: 500")
        assert err.detail == "Internal: 500"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ExtractionError) as exc_info:
            raise ExtractionError(user_message="boom")
        assert exc_info.value.user_message == "boom"


class TestErrorSubclasses:
    @pytest.mark.parametrize(
        "error_class",
        [
            APIUnavailableError,
            RateLimitError,
            TokenLimitError,
            NetworkError,
            ParsingError,
            UnknownAPIError,
            MissingAPIKeyError,
        ],
    )
    def test_inherits_from_extraction_error(self, error_class):
        err = error_class(user_message="test")
        assert isinstance(err, ExtractionError)

    @pytest.mark.parametrize(
        "error_class",
        [
            APIUnavailableError,
            RateLimitError,
            TokenLimitError,
            NetworkError,
            ParsingError,
            UnknownAPIError,
            MissingAPIKeyError,
        ],
    )
    def test_can_be_raised_and_caught_as_extraction_error(self, error_class):
        with pytest.raises(ExtractionError):
            raise error_class(user_message="test message")

    def test_missing_api_key_error(self):
        err = MissingAPIKeyError(user_message="No API key")
        assert err.user_message == "No API key"

    def test_rate_limit_error_with_detail(self):
        err = RateLimitError(user_message="Rate limited", detail="429 Too Many Requests")
        assert err.detail == "429 Too Many Requests"


class TestDefaultUserMessages:
    def test_returns_dict(self):
        messages = default_user_messages()
        assert isinstance(messages, dict)

    def test_all_expected_keys_present(self):
        messages = default_user_messages()
        expected_keys = {
            "missing_api_key",
            "api_unavailable",
            "rate_limited",
            "token_limit",
            "network",
            "parsing",
            "unknown",
        }
        assert expected_keys == set(messages.keys())

    def test_all_values_are_non_empty_strings(self):
        messages = default_user_messages()
        for key, value in messages.items():
            assert isinstance(value, str), f"Message for '{key}' is not a string"
            assert value.strip(), f"Message for '{key}' is empty"

    def test_missing_api_key_message_mentions_api_key(self):
        messages = default_user_messages()
        assert "api key" in messages["missing_api_key"].lower()

    def test_rate_limited_message_mentions_wait(self):
        messages = default_user_messages()
        assert "wait" in messages["rate_limited"].lower()
