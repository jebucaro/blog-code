import re

import pytest

from nodus.extractor import SYSTEM_PROMPT, TRANSCRIPT_SYSTEM_PROMPT, build_system_prompt
from nodus.models import (
    DOCUMENT_NODE_TYPES,
    RELATIONSHIP_TYPE_EXAMPLES,
    SPEAKER_LINK_TYPES,
    TRANSCRIPT_NODE_TYPES,
    TRANSCRIPT_RELATIONSHIP_TYPE_EXAMPLES,
)

NEW_NODE_TYPES = ["topic", "statement", "decision", "action_item"]
NEW_REL_TYPES = TRANSCRIPT_RELATIONSHIP_TYPE_EXAMPLES[len(RELATIONSHIP_TYPE_EXAMPLES):]


def has_token(text: str, token: str) -> bool:
    """Whole-token match: 'statement' must not match 'restatement'."""
    return re.search(rf"(?<![A-Za-z_]){re.escape(token)}(?![A-Za-z_])", text) is not None


class TestSystemPrompt:
    def test_placeholders_are_formatted(self):
        assert "{node_types}" not in SYSTEM_PROMPT
        assert "{rel_types}" not in SYSTEM_PROMPT

    def test_controlled_vocabulary_present(self):
        for node_type in DOCUMENT_NODE_TYPES:
            assert node_type in SYSTEM_PROMPT
        for rel_type in RELATIONSHIP_TYPE_EXAMPLES:
            assert rel_type in SYSTEM_PROMPT

    def test_final_self_check_present(self):
        assert "Final Self-Check" in SYSTEM_PROMPT
        assert "source_node_id" in SYSTEM_PROMPT
        assert "target_node_id" in SYSTEM_PROMPT

    def test_security_rules_preserved(self):
        assert "CRITICAL SECURITY RULES" in SYSTEM_PROMPT

    def test_no_orphan_subsections(self):
        # The old scrambled 3b/4b/5b sections must be gone.
        assert "3b." not in SYSTEM_PROMPT
        assert "4b." not in SYSTEM_PROMPT
        assert "5b." not in SYSTEM_PROMPT

    def test_context_field_guidance_present_for_nodes_and_relationships(self):
        assert SYSTEM_PROMPT.count("`context` (Supporting Detail - always present, may be empty)") == 2
        assert "Never restate the `label` or `type`" in SYSTEM_PROMPT

    def test_context_guidance_actively_asks_for_detail_not_just_permits_it(self):
        assert "Most entities in a text with real detail" in SYSTEM_PROMPT
        assert "Most relationships in a text with real detail" in SYSTEM_PROMPT

    def test_context_guidance_says_use_an_empty_string_when_nothing_to_add(self):
        assert SYSTEM_PROMPT.count("Use an empty string only when") >= 2

    def test_final_self_check_covers_context(self):
        self_check = SYSTEM_PROMPT.split("Final Self-Check")[1]
        assert "`context`" in self_check


class TestBuildSystemPrompt:
    def test_document_prompt_is_the_module_constant(self):
        assert build_system_prompt("document") == SYSTEM_PROMPT

    def test_transcript_prompt_is_the_module_constant(self):
        assert build_system_prompt("transcript") == TRANSCRIPT_SYSTEM_PROMPT

    def test_unknown_input_type_raises(self):
        with pytest.raises(ValueError):
            build_system_prompt("meeting")

    def test_document_prompt_has_no_transcript_vocabulary(self):
        for token in [*NEW_NODE_TYPES, *NEW_REL_TYPES]:
            assert not has_token(SYSTEM_PROMPT, token), token
        assert "Transcripts and conversations" not in SYSTEM_PROMPT


class TestTranscriptPrompt:
    def test_no_unformatted_placeholders(self):
        assert "{node_types}" not in TRANSCRIPT_SYSTEM_PROMPT
        assert "{rel_types}" not in TRANSCRIPT_SYSTEM_PROMPT

    def test_security_rules_preserved(self):
        assert "CRITICAL SECURITY RULES" in TRANSCRIPT_SYSTEM_PROMPT
        assert "UNTRUSTED" in TRANSCRIPT_SYSTEM_PROMPT

    def test_offers_transcript_vocabulary(self):
        for token in [*TRANSCRIPT_NODE_TYPES, *TRANSCRIPT_RELATIONSHIP_TYPE_EXAMPLES]:
            assert has_token(TRANSCRIPT_SYSTEM_PROMPT, token), token

    def test_has_conversation_section(self):
        assert "## Transcripts and conversations" in TRANSCRIPT_SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "phrase",
        [
            "speaker not identified",
            "Unknown speaker",
            "Speaker 2",
            "timestamp",
            "10 to 30",
            "small talk",
            "do not guess",
            "mis-transcri",
        ],
    )
    def test_key_instruction_present(self, phrase):
        assert phrase.lower() in TRANSCRIPT_SYSTEM_PROMPT.lower()

    def test_first_person_pronouns_are_addressed(self):
        for pronoun in ('"I"', '"we"', '"you"'):
            assert pronoun in TRANSCRIPT_SYSTEM_PROMPT

    def test_first_person_pronouns_need_a_known_speaker(self):
        flat = " ".join(TRANSCRIPT_SYSTEM_PROMPT.split())
        sentences = [x for x in re.split(r"(?<=[.;:])\s", flat) if '"I"' in x]
        assert sentences
        assert all("only when the speaker of that passage is known" in x for x in sentences), sentences

    def test_transcript_prompt_handles_non_conversation_text(self):
        assert "not a conversation" in TRANSCRIPT_SYSTEM_PROMPT.lower()

    def test_self_check_covers_transcript_rules(self):
        self_check = TRANSCRIPT_SYSTEM_PROMPT.split("Final Self-Check")[1]
        assert has_token(self_check, "ABOUT")
        assert "speaker not identified" in self_check

    def test_role_fallback_names_host_and_guest(self):
        assert '("Host", "Guest"' in TRANSCRIPT_SYSTEM_PROMPT

    def test_self_check_lists_every_speaker_link_type(self):
        self_check = TRANSCRIPT_SYSTEM_PROMPT.split("Final Self-Check")[1]
        for link in SPEAKER_LINK_TYPES:
            assert has_token(self_check, link), link

    def test_about_is_not_called_a_speaker_link(self):
        # No sentence that mentions "speaker link(s)" may also contain the token ABOUT.
        text = " ".join(TRANSCRIPT_SYSTEM_PROMPT.split())
        for sentence in re.split(r"(?<=[.;:])\s", text):
            if "speaker link" in sentence.lower():
                assert not has_token(sentence, "ABOUT"), sentence

    def test_marker_rule_is_limited_to_statement_decision_action_item(self):
        assert "Every node whose speaker" not in TRANSCRIPT_SYSTEM_PROMPT
        self_check = " ".join(TRANSCRIPT_SYSTEM_PROMPT.split("Final Self-Check")[1].split())
        sentence = next(x for x in self_check.split(". ") if "speaker not identified" in x)
        for node_type in ("statement", "decision", "action_item"):
            assert node_type in sentence

    def test_turn_order_is_not_a_cue(self):
        flat = " ".join(TRANSCRIPT_SYSTEM_PROMPT.split())
        sentences = [
            x for x in re.split(r"(?<=[.;:])\s", flat) if "order of turns" in x.lower()
        ]
        assert sentences
        assert all("NOT a cue" in x for x in sentences), sentences

    def test_importance_overrides_completeness(self):
        section = TRANSCRIPT_SYSTEM_PROMPT.split("## Transcripts and conversations")[1]
        section = section.split("## 3. Core Principles")[0]
        flat = " ".join(section.split())
        sentences = re.split(r"(?<=[.;:])\s", flat)
        assert any("override" in x and "Completeness" in x for x in sentences)

    def test_unknown_speaker_label_is_not_a_participant(self):
        flat = " ".join(TRANSCRIPT_SYSTEM_PROMPT.split())
        assert "is not a participant" in flat
