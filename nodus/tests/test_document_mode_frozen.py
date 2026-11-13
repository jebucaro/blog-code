import json
from pathlib import Path

from nodus.extractor import SYSTEM_PROMPT
from nodus.models import extraction_json_schema

FIXTURES = Path(__file__).parent / "fixtures"


def test_document_system_prompt_is_unchanged():
    expected = (FIXTURES / "document_system_prompt.txt").read_text(encoding="utf-8")
    assert SYSTEM_PROMPT == expected


def test_document_extraction_schema_is_unchanged():
    expected = json.loads(
        (FIXTURES / "document_extraction_schema.json").read_text(encoding="utf-8")
    )
    assert extraction_json_schema() == expected
