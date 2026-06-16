"""Evidence extraction using keyword matching and multiple similarity scores."""

import difflib
import logging
import math
import re
from collections import Counter

from utils import split_sentences, tokenize_words


LOGGER = logging.getLogger(__name__)


class EvidenceExtractor:
    """Rank sentences from documents as candidate evidence."""

    def extract(self, document_text, keywords, claim, max_sentences=5, claim_representation=None):
        """Return top matching evidence sentences."""
        sentences = split_sentences(document_text)
        if not sentences:
            return []
        claim_tokens = set(tokenize_words(claim))
        keyword_set = set(token.lower() for token in keywords)
        scored = []
        for index, sentence in enumerate(sentences):
            score, similarity_details = self.relevance_score(sentence, keyword_set, claim_tokens, claim, claim_representation)
            if score > 0:
                scored.append(
                    {
                        "sentence": sentence,
                        "context": self.context_window(sentences, index),
                        "relevance": round(score, 4),
                        "claim_coverage": self.claim_coverage(sentence, claim_tokens),
                        "jaccard_similarity": similarity_details["jaccard_similarity"],
                        "cosine_similarity": similarity_details["cosine_similarity"],
                        "sequence_similarity": similarity_details["sequence_similarity"],
                        "combined_similarity": similarity_details["combined_similarity"],
                        "semantic_score": similarity_details["semantic_score"],
                        "entity_overlap": similarity_details["entity_overlap"],
                        "subject_match": similarity_details["subject_match"],
                        "predicate_match": similarity_details["predicate_match"],
                        "object_match": similarity_details["object_match"],
                        "number_match": similarity_details["number_match"],
                        "date_match": similarity_details["date_match"],
                    }
                )
        scored.sort(key=lambda item: item["relevance"], reverse=True)
        return scored[:max_sentences]

    def relevance_score(self, sentence, keyword_set, claim_tokens, claim, claim_representation=None):
        """Score a sentence using keyword coverage, token overlap, and difflib."""
        sentence_tokens = tokenize_words(sentence)
        if not sentence_tokens:
            return 0.0, {
                "jaccard_similarity": 0.0,
                "cosine_similarity": 0.0,
                "sequence_similarity": 0.0,
                "combined_similarity": 0.0,
                "semantic_score": 0,
                "entity_overlap": 0,
                "subject_match": False,
                "predicate_match": False,
                "object_match": False,
                "number_match": False,
                "date_match": False,
            }
        counts = Counter(sentence_tokens)
        keyword_hits = sum(counts.get(keyword, 0) for keyword in keyword_set if " " not in keyword)
        phrase_hits = sum(1 for keyword in keyword_set if " " in keyword and keyword in sentence.lower())
        overlap = len(set(sentence_tokens) & claim_tokens)
        sequence_similarity = difflib.SequenceMatcher(None, sentence.lower(), (claim or "").lower()).ratio()
        jaccard_similarity = self.jaccard_similarity(sentence_tokens, claim_tokens)
        cosine_similarity = self.cosine_similarity(sentence_tokens, claim_tokens)
        coverage = overlap / max(1, len(claim_tokens))
        semantic = self.semantic_features(sentence, sentence_tokens, claim_representation or {})
        combined_similarity = round(((jaccard_similarity + cosine_similarity + sequence_similarity) / 3) * 100, 2)
        score = (
            (keyword_hits * 1.5)
            + (phrase_hits * 2.0)
            + (overlap * 0.8)
            + (sequence_similarity * 2.0)
            + (jaccard_similarity * 4.0)
            + (cosine_similarity * 4.0)
            + (coverage * 3.0)
            + (semantic["semantic_score"] / 100 * 4.0)
        )
        return score, {
            "jaccard_similarity": round(jaccard_similarity * 100, 2),
            "cosine_similarity": round(cosine_similarity * 100, 2),
            "sequence_similarity": round(sequence_similarity * 100, 2),
            "combined_similarity": combined_similarity,
            **semantic,
        }

    def semantic_features(self, sentence, sentence_tokens, claim_representation):
        """Return interpretable semantic matching features for the sentence."""
        sentence_set = set(sentence_tokens)
        subject_tokens = set(tokenize_words(claim_representation.get("subject", "")))
        predicate_tokens = set(tokenize_words(claim_representation.get("relation", "") or claim_representation.get("predicate", "")))
        object_tokens = set(tokenize_words(claim_representation.get("object", "")))
        entity_terms = set()
        for value in claim_representation.get("people", []) + claim_representation.get("organizations", []) + claim_representation.get("locations", []):
            entity_terms.update(tokenize_words(value))
        claim_numbers = set(claim_representation.get("numbers", []))
        claim_dates = set(claim_representation.get("dates", []))
        sentence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", sentence or ""))
        sentence_dates = set(re.findall(r"\b(?:today|yesterday|tomorrow|(?:18|19|20)\d{2}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b", sentence or "", flags=re.IGNORECASE))
        subject_match = bool(subject_tokens and subject_tokens & sentence_set)
        predicate_match = bool(predicate_tokens and predicate_tokens & sentence_set)
        object_match = bool(object_tokens and object_tokens & sentence_set)
        entity_overlap = len(entity_terms & sentence_set)
        number_match = bool(claim_numbers and claim_numbers & sentence_numbers)
        date_match = bool(claim_dates and {value.lower() for value in claim_dates} & {value.lower() for value in sentence_dates})
        semantic_score = 0
        semantic_score += 20 if subject_match else 0
        semantic_score += 20 if predicate_match else 0
        semantic_score += 20 if object_match else 0
        semantic_score += min(20, entity_overlap * 10)
        semantic_score += 10 if number_match else 0
        semantic_score += 10 if date_match else 0
        return {
            "semantic_score": min(100, semantic_score),
            "entity_overlap": entity_overlap,
            "subject_match": subject_match,
            "predicate_match": predicate_match,
            "object_match": object_match,
            "number_match": number_match,
            "date_match": date_match,
        }

    def jaccard_similarity(self, sentence_tokens, claim_tokens):
        """Return Jaccard similarity between two token sets."""
        sentence_set = set(sentence_tokens)
        claim_set = set(claim_tokens)
        union = sentence_set | claim_set
        if not union:
            return 0.0
        return len(sentence_set & claim_set) / len(union)

    def cosine_similarity(self, sentence_tokens, claim_tokens):
        """Return cosine similarity between two bag-of-words vectors."""
        sentence_counts = Counter(sentence_tokens)
        claim_counts = Counter(claim_tokens)
        if not sentence_counts or not claim_counts:
            return 0.0
        common = set(sentence_counts) & set(claim_counts)
        numerator = sum(sentence_counts[token] * claim_counts[token] for token in common)
        sentence_norm = math.sqrt(sum(count * count for count in sentence_counts.values()))
        claim_norm = math.sqrt(sum(count * count for count in claim_counts.values()))
        denominator = sentence_norm * claim_norm
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def claim_coverage(self, sentence, claim_tokens):
        """Return percentage of claim terms covered by the evidence sentence."""
        sentence_tokens = set(tokenize_words(sentence))
        return int(round((len(sentence_tokens & claim_tokens) / max(1, len(claim_tokens))) * 100))

    def context_window(self, sentences, index):
        """Return nearby text for transparent evidence inspection."""
        start = max(0, index - 1)
        end = min(len(sentences), index + 2)
        return " ".join(sentences[start:end])
