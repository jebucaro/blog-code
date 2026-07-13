from nodus.extractor import SYSTEM_PROMPT
from nodus.models import NODE_TYPES, RELATIONSHIP_TYPE_EXAMPLES


class TestSystemPrompt:
    def test_placeholders_are_formatted(self):
        assert "{node_types}" not in SYSTEM_PROMPT
        assert "{rel_types}" not in SYSTEM_PROMPT

    def test_controlled_vocabulary_present(self):
        for node_type in NODE_TYPES:
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
