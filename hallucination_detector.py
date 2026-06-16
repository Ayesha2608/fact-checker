"""Compare generated explanations against extracted evidence."""

import difflib

from utils import split_sentences, tokenize_words


class HallucinationDetector:
    """Flag answer statements that are not grounded in evidence."""

    def analyze(self, answer_text, evidence_items, min_similarity=0.22):
        """Return unsupported statements from a generated answer."""
        evidence_sentences = [item.get("sentence", "") for item in evidence_items]
        unsupported = []
        for statement in split_sentences(answer_text):
            if not self._is_supported(statement, evidence_sentences, min_similarity):
                unsupported.append(statement)
        return {
            "unsupported_statements": unsupported,
            "warning": bool(unsupported),
            "message": "Unsupported claims detected" if unsupported else "All answer statements are evidence-aligned",
        }

    def _is_supported(self, statement, evidence_sentences, min_similarity):
        statement_tokens = set(tokenize_words(statement))
        if not statement_tokens:
            return True
        for evidence in evidence_sentences:
            evidence_tokens = set(tokenize_words(evidence))
            overlap = len(statement_tokens & evidence_tokens) / max(1, len(statement_tokens))
            similarity = difflib.SequenceMatcher(None, statement.lower(), evidence.lower()).ratio()
            if overlap >= 0.35 or similarity >= min_similarity:
                return True
        return False
