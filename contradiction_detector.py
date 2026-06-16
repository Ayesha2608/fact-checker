"""Rule-based contradiction detection for claim verification."""

import re

from utils import tokenize_words


class ContradictionDetector:
    """Detect obvious contradiction signals without machine learning."""

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
        "from", "has", "have", "had", "in", "into", "is", "it", "its", "of",
        "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
    }

    NEGATION_WORDS = {
        "not", "no", "never", "none", "cannot", "can't", "isn't", "aren't", "wasn't",
        "weren't", "didn't", "doesn't", "false", "denied", "reject", "rejected",
        "incorrect", "failed", "fake", "hoax", "misleading", "debunked", "myth",
    }

    SUPPORT_WORDS = {
        "confirmed", "true", "accurate", "approved", "verified", "announced",
        "reported", "found", "shows", "according", "evidence", "yes",
    }
    DEBUNK_WORDS = {
        "myth", "misconception", "false", "incorrect", "debunked", "not true",
        "fact check", "hoax", "misleading",
    }
    ANTONYM_PAIRS = {
        ("increase", "decrease"), ("increased", "decreased"), ("rise", "fall"),
        ("rises", "falls"), ("safe", "unsafe"), ("effective", "ineffective"),
        ("legal", "illegal"), ("possible", "impossible"), ("allow", "ban"),
        ("allowed", "banned"), ("win", "lose"), ("won", "lost"),
    }
    NUMBER_WORDS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    DEFINITION_CONFLICTS = {
        "moon": {"quasi", "quasi-moon", "asteroid", "temporary"},
        "moons": {"quasi", "quasi-moon", "asteroid", "temporary"},
        "satellite": {"asteroid", "quasi"},
    }

    def classify(self, claim, sentence, claim_representation=None):
        """Classify a sentence as support, contradiction, or neutral."""
        claim_tokens = self._content_tokens(claim)
        sentence_tokens = self._content_tokens(sentence)
        if not sentence_tokens:
            return "neutral"
        overlap = len(claim_tokens & sentence_tokens)
        if overlap == 0:
            return "neutral"
        overlap_ratio = overlap / max(1, len(claim_tokens))
        claim_negative = bool(claim_tokens & self.NEGATION_WORDS)
        sentence_negative = bool(sentence_tokens & self.NEGATION_WORDS)
        if self._date_or_number_mismatch(claim, sentence, overlap):
            return "contradict"
        if self._debunk_framing(claim_tokens, sentence):
            return "contradict"
        if self._definition_conflict(claim_tokens, sentence_tokens):
            return "contradict"
        if claim_representation and self._entity_conflict(sentence_tokens, claim_representation, overlap_ratio):
            return "contradict"
        if self._antonym_mismatch(claim_tokens, sentence_tokens):
            return "contradict"
        if sentence_negative and not claim_negative and self._negation_targets_claim(claim_tokens, sentence):
            return "contradict"
        if claim_negative and not sentence_negative and overlap >= 2:
            return "contradict"
        if sentence_tokens & self.SUPPORT_WORDS and overlap_ratio >= 0.35:
            return "support"
        if claim_representation and self._semantic_support(sentence_tokens, claim_representation, overlap_ratio):
            return "support"
        if overlap_ratio >= 0.55 or (len(claim_tokens) <= 4 and overlap_ratio >= 0.5):
            return "support"
        return "neutral"

    def has_contradiction_terms(self, text):
        """Return True if text contains direct contradiction vocabulary."""
        return bool(set(tokenize_words(text)) & self.NEGATION_WORDS)

    def _date_or_number_mismatch(self, claim, sentence, overlap):
        """Detect direct year or quantity conflicts when the texts otherwise overlap."""
        if overlap < 1:
            return False
        claim_years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", claim or ""))
        sentence_years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", sentence or ""))
        if claim_years and sentence_years and claim_years.isdisjoint(sentence_years):
            return True
        claim_numbers = self._important_numbers(claim)
        sentence_numbers = self._important_numbers(sentence)
        if claim_numbers and sentence_numbers and claim_numbers.isdisjoint(sentence_numbers):
            return True
        return False

    def _important_numbers(self, text):
        """Extract quantities while avoiding embedded product/disease codes."""
        numbers = set()
        tokens = tokenize_words(text)
        for token in tokens:
            if token in self.NUMBER_WORDS:
                numbers.add(self.NUMBER_WORDS[token])
        for match in re.finditer(r"\b\d+(?:\.\d+)?%?\b", text or ""):
            start, end = match.span()
            before = (text or "")[max(0, start - 2):start]
            after = (text or "")[end:end + 2]
            if "-" in before or "-" in after:
                continue
            numbers.add(match.group(0))
        return numbers

    def _definition_conflict(self, claim_tokens, sentence_tokens):
        for term, conflicts in self.DEFINITION_CONFLICTS.items():
            if term in claim_tokens and conflicts & sentence_tokens:
                return True
        return False

    def _debunk_framing(self, claim_tokens, sentence):
        lower = (sentence or "").lower()
        sentence_tokens = self._content_tokens(sentence)
        overlap_ratio = len(claim_tokens & sentence_tokens) / max(1, len(claim_tokens))
        if overlap_ratio < 0.35:
            return False
        if any(word in lower for word in self.DEBUNK_WORDS):
            return True
        if "fact:" in lower and bool(claim_tokens & self.NEGATION_WORDS):
            positive_markers = {"can", "does", "do", "often", "repeatedly", "multiple", "many", "is", "are"}
            if set(tokenize_words(sentence)) & positive_markers:
                return True
        return False

    def _entity_conflict(self, sentence_tokens, claim_representation, overlap_ratio):
        if overlap_ratio < 0.35:
            return False
        locations = set()
        for location in claim_representation.get("locations", []):
            locations.update(tokenize_words(location))
        if not locations:
            return False
        alternative_locations = {
            "pakistan", "india", "china", "russia", "ukraine", "israel", "iran",
            "iraq", "syria", "france", "germany", "canada", "australia", "japan",
            "turkey", "karachi", "lahore", "islamabad", "delhi", "mumbai",
        }
        mentioned = sentence_tokens & alternative_locations
        return bool(mentioned and mentioned.isdisjoint(locations))

    def _semantic_support(self, sentence_tokens, claim_representation, overlap_ratio):
        subject = set(tokenize_words(claim_representation.get("subject", "")))
        relation = set(tokenize_words(claim_representation.get("relation", "") or claim_representation.get("predicate", "")))
        obj = set(tokenize_words(claim_representation.get("object", "")))
        locations = set()
        for location in claim_representation.get("locations", []):
            locations.update(tokenize_words(location))
        core_hits = 0
        core_hits += 1 if subject and subject & sentence_tokens else 0
        core_hits += 1 if relation and relation & sentence_tokens else 0
        core_hits += 1 if obj and obj & sentence_tokens else 0
        core_hits += 1 if locations and locations & sentence_tokens else 0
        return core_hits >= 2 and overlap_ratio >= 0.35

    def _content_tokens(self, text):
        """Return meaningful tokens for stance comparison."""
        return {
            token
            for token in tokenize_words(text)
            if token not in self.STOPWORDS and (len(token) > 1 or token.isdigit())
        }

    def _negation_targets_claim(self, claim_tokens, sentence):
        """Check whether a negation word locally negates claim terms."""
        tokens = tokenize_words(sentence)
        negations = self.NEGATION_WORDS
        for index, token in enumerate(tokens):
            if token not in negations:
                continue
            if token in {"not", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "isn", "aren", "wasn"} and index + 1 < len(tokens) and tokens[index + 1] in {"only", "just", "necessarily"}:
                continue
            if token in {"isn", "aren", "wasn"} and index + 2 < len(tokens) and tokens[index + 1] == "t" and tokens[index + 2] in {"only", "just", "necessarily"}:
                continue
            scope = {
                value for value in tokens[index + 1:index + 7]
                if value not in self.STOPWORDS
            }
            if scope & claim_tokens:
                return True
            if token in {"false", "fake", "hoax", "debunked", "misleading", "incorrect"} and len(scope | claim_tokens) > 0:
                return len(claim_tokens & self._content_tokens(sentence)) / max(1, len(claim_tokens)) >= 0.5
        return False

    def _antonym_mismatch(self, claim_tokens, sentence_tokens):
        for first, second in self.ANTONYM_PAIRS:
            if first in claim_tokens and second in sentence_tokens:
                return True
            if second in claim_tokens and first in sentence_tokens:
                return True
        return False
