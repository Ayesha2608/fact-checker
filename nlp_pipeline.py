"""A transparent NLP pipeline implemented with Python Standard Library only."""

import re
from collections import Counter

from utils import normalize_space, split_sentences, tokenize_words


class StandardLibraryNLPPipeline:
    """Run basic and higher-level NLP stages without third-party packages."""

    QUESTION_WORDS = {"what", "when", "where", "who", "why", "how", "did", "does", "is", "are", "was", "were"}
    NEGATIONS = {"not", "no", "never", "cannot", "can't", "isn't", "wasn't", "denied", "false"}
    MODALS = {"may", "might", "could", "should", "would", "can", "must", "will"}
    DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
    PREPOSITIONS = {
        "about", "above", "after", "against", "around", "at", "before", "between",
        "by", "during", "for", "from", "in", "into", "near", "of", "on", "over",
        "through", "to", "under", "with", "without",
    }
    VERB_HINTS = {
        "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
        "do", "does", "did", "say", "says", "said", "claim", "claims", "claimed",
        "attack", "attacked", "happen", "happened", "declare", "declared", "show",
        "shows", "report", "reported", "confirm", "confirmed", "deny", "denied",
        "reduce", "reduced", "increase", "increased", "cause", "caused", "prove",
        "proved", "find", "found", "announce", "announced", "strike", "struck",
        "kill", "killed", "killing", "injure", "injured", "injuring",
    }
    DISCOURSE_MARKERS = {
        "because", "however", "therefore", "although", "but", "while", "after",
        "before", "according", "despite", "since",
    }
    COUNTRIES = {
        "pakistan", "india", "china", "russia", "ukraine", "israel", "lebanon", "iran",
        "iraq", "syria", "palestine", "united states", "america", "usa", "france",
        "germany", "canada", "australia", "japan", "turkey", "china",
    }
    CITIES = {
        "beirut", "karachi", "lahore", "islamabad", "delhi", "mumbai", "new york",
        "washington", "london", "paris", "tokyo", "gaza", "jerusalem", "moscow",
        "qinghai",
    }
    DEMONYMS = {"chinese", "pakistani", "indian", "american", "russian", "ukrainian", "iranian", "afghan"}
    ORGANIZATION_ACRONYMS = {
        "WHO", "UN", "NATO", "EU", "FBI", "CIA", "CDC", "NASA", "IMF", "WTO",
        "UNICEF", "UNESCO", "BBC", "CNN", "AP",
    }
    HEALTH_TERMS = {"COVID-19", "COVID", "SARS-COV-2", "HIV", "AIDS"}
    CELESTIAL_TERMS = {
        "earth", "moon", "mars", "venus", "jupiter", "saturn", "mercury",
        "uranus", "neptune", "sun",
    }
    IRREGULAR_LEMMAS = {
        "was": "be", "were": "be", "is": "be", "are": "be", "been": "be",
        "has": "have", "had": "have", "does": "do", "did": "do",
        "said": "say", "says": "say", "found": "find", "children": "child",
        "men": "man", "women": "woman", "countries": "country",
    }

    def __init__(self, stopwords=None):
        self.stopwords = set(stopwords or [])

    def process(self, text):
        """Run the complete NLP pipeline and return all intermediate artifacts."""
        raw_text = text or ""
        normalized = normalize_space(raw_text)
        sentences = split_sentences(normalized) or ([normalized] if normalized else [])
        raw_tokens = tokenize_words(normalized)
        filtered_tokens = [token for token in raw_tokens if token not in self.stopwords and len(token) > 1]
        stems = [self.stem(token) for token in filtered_tokens]
        lemmas = [self.lemmatize(token) for token in filtered_tokens]
        pos_tags = self.pos_tag(raw_tokens)
        entities = self.named_entities(normalized)
        ngrams = self.ngrams(filtered_tokens)
        noun_phrases = self.phrase_chunks(pos_tags, target="noun")
        verb_phrases = self.phrase_chunks(pos_tags, target="verb")
        claim_structure = self.claim_structure(filtered_tokens, pos_tags, entities)
        return {
            "stages": [
                "text_normalization",
                "sentence_segmentation",
                "tokenization",
                "stopword_removal",
                "stemming",
                "lemmatization",
                "ngram_generation",
                "pos_tagging",
                "named_entity_recognition",
                "phrase_chunking",
                "claim_structure_analysis",
                "negation_scope_detection",
                "semantic_role_heuristics",
            ],
            "raw_text": raw_text,
            "normalized_text": normalized,
            "sentences": sentences,
            "sentence_count": len(sentences),
            "raw_tokens": raw_tokens,
            "tokens": filtered_tokens,
            "token_count": len(filtered_tokens),
            "stems": stems,
            "lemmas": lemmas,
            "term_frequency": dict(Counter(filtered_tokens)),
            "ngrams": ngrams,
            "pos_tags": pos_tags,
            "named_entities": entities,
            "noun_phrases": noun_phrases,
            "verb_phrases": verb_phrases,
            "claim_structure": claim_structure,
            "negation_scope": self.negation_scope(raw_tokens),
            "discourse_markers": [token for token in raw_tokens if token in self.DISCOURSE_MARKERS],
            "lexical_diversity": self.lexical_diversity(filtered_tokens),
            "lexical_density": self.lexical_density(pos_tags),
            "readability": self.readability(sentences, raw_tokens),
        }

    def stem(self, token):
        """Apply a small rule-based stemmer for project transparency."""
        for suffix in ("ization", "ational", "fulness", "iveness", "ingly", "edly", "ment", "tion", "ing", "ied", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                if suffix == "ied":
                    return token[: -len(suffix)] + "y"
                return token[: -len(suffix)]
        return token

    def lemmatize(self, token):
        """Return a small rule-based lemma for common English inflections."""
        if token in self.IRREGULAR_LEMMAS:
            return self.IRREGULAR_LEMMAS[token]
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ves") and len(token) > 4:
            return token[:-3] + "f"
        if token.endswith("ing") and len(token) > 5:
            base = token[:-3]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if token.endswith("ed") and len(token) > 4:
            base = token[:-2]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token

    def ngrams(self, tokens):
        """Return bigrams and trigrams for phrase-level NLP features."""
        return {
            "bigrams": [" ".join(tokens[index:index + 2]) for index in range(max(0, len(tokens) - 1))],
            "trigrams": [" ".join(tokens[index:index + 3]) for index in range(max(0, len(tokens) - 2))],
        }

    def pos_tag(self, raw_tokens):
        """Assign heuristic part-of-speech tags using word-shape and suffix rules."""
        tags = []
        for index, token in enumerate(raw_tokens):
            tag = "NOUN"
            if token in self.NEGATIONS:
                tag = "NEG"
            elif token in self.MODALS:
                tag = "MODAL"
            elif token in self.DETERMINERS:
                tag = "DET"
            elif token in self.PREPOSITIONS:
                tag = "ADP"
            elif token in self.QUESTION_WORDS:
                tag = "PRON"
            elif token in self.VERB_HINTS or token.endswith(("ed", "ing", "ize", "ise")):
                tag = "VERB"
            elif token.endswith("ly"):
                tag = "ADV"
            elif token.endswith(("ous", "ive", "al", "ic", "able", "ible")):
                tag = "ADJ"
            elif re.fullmatch(r"\d+(?:\.\d+)?", token):
                tag = "NUM"
            elif index > 0 and raw_tokens[index - 1] in self.DETERMINERS:
                tag = "NOUN"
            tags.append({"token": token, "tag": tag})
        return tags

    def named_entities(self, text):
        """Detect people, organizations, countries, cities, dates, and numbers."""
        original = text or ""
        lower = original.lower()
        dates = re.findall(
            r"\b(?:today|yesterday|tomorrow|(?:18|19|20)\d{2}|"
            r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
            original,
            flags=re.IGNORECASE,
        )
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", original)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", original)
        acronyms = re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", original)
        countries = sorted({name.title() for name in self.COUNTRIES if re.search(r"\b" + re.escape(name) + r"\b", lower)})
        cities = sorted({name.title() for name in self.CITIES if re.search(r"\b" + re.escape(name) + r"\b", lower)})
        people = []
        organizations = []
        medical_terms = []
        organization_markers = {"Organization", "University", "Agency", "Ministry", "Court", "Reuters", "BBC", "WHO", "UN", "NATO"}
        for item in capitalized:
            item_lower = item.lower()
            if item_lower in self.COUNTRIES or item_lower in self.CITIES or item_lower in self.QUESTION_WORDS or item_lower in self.CELESTIAL_TERMS or item_lower in self.DEMONYMS:
                continue
            if any(marker in item for marker in organization_markers):
                organizations.append(item)
            else:
                people.append(item)
        for acronym in acronyms:
            if acronym in self.ORGANIZATION_ACRONYMS:
                organizations.append(acronym)
            elif acronym in self.HEALTH_TERMS or re.search(r"\d", acronym):
                medical_terms.append(acronym)
        return {
            "people": self.unique(people),
            "organizations": self.unique(organizations),
            "countries": countries,
            "cities": cities,
            "dates": self.unique(dates),
            "numbers": self.unique(numbers),
            "medical_terms": self.unique(medical_terms),
        }

    def phrase_chunks(self, pos_tags, target):
        """Build simple noun or verb phrase chunks from POS tags."""
        chunks = []
        current = []
        allowed = {"noun": {"NOUN", "ADJ", "NUM"}, "verb": {"VERB", "ADV", "MODAL", "NEG"}}[target]
        for item in pos_tags:
            if item["tag"] in allowed:
                current.append(item["token"])
            else:
                if current:
                    chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return [chunk for chunk in chunks if len(chunk.split()) <= 5][:10]

    def claim_structure(self, tokens, pos_tags, entities):
        """Extract a transparent subject-predicate-object style claim frame."""
        predicate_index = None
        for index, item in enumerate(pos_tags):
            if item["tag"] == "VERB":
                predicate_index = index
                break
        predicate = pos_tags[predicate_index]["token"] if predicate_index is not None else ""
        subject = ""
        obj = ""
        if predicate_index is not None:
            before = [item["token"] for item in pos_tags[:predicate_index] if item["tag"] in {"NOUN", "ADJ", "NUM"}]
            after = [item["token"] for item in pos_tags[predicate_index + 1:] if item["tag"] in {"NOUN", "ADJ", "NUM"}]
            subject = " ".join(before[-4:])
            obj = " ".join(after[:6])
        actors = entities.get("people", []) + entities.get("organizations", [])
        locations = entities.get("countries", []) + entities.get("cities", [])
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "actors": actors,
            "locations": locations,
            "has_negation": bool(set(tokens) & self.NEGATIONS),
            "modality": [token for token in tokens if token in self.MODALS],
        }

    def negation_scope(self, raw_tokens):
        """Capture the local token window after negation words."""
        scopes = []
        for index, token in enumerate(raw_tokens):
            if token in self.NEGATIONS:
                scopes.append({"negation": token, "scope": raw_tokens[index + 1:index + 5]})
        return scopes

    def lexical_diversity(self, tokens):
        """Return type-token ratio as a percentage."""
        if not tokens:
            return 0
        return int(round((len(set(tokens)) / len(tokens)) * 100))

    def lexical_density(self, pos_tags):
        """Return percentage of content words based on heuristic POS tags."""
        if not pos_tags:
            return 0
        content_tags = {"NOUN", "VERB", "ADJ", "ADV", "NUM"}
        content = len([item for item in pos_tags if item["tag"] in content_tags])
        return int(round((content / len(pos_tags)) * 100))

    def readability(self, sentences, raw_tokens):
        """Return simple readability proxies for viva explanation."""
        average_sentence_length = 0
        if sentences:
            average_sentence_length = round(len(raw_tokens) / max(1, len(sentences)), 2)
        return {
            "average_sentence_length": average_sentence_length,
            "long_sentence_count": len([sentence for sentence in sentences if len(tokenize_words(sentence)) > 25]),
        }

    def unique(self, values):
        seen = set()
        output = []
        for value in values:
            cleaned = (value or "").strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                output.append(cleaned)
        return output
