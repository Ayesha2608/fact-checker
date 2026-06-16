"""Query processing: tokenization, stopword removal, intent detection."""

import logging
import re
from collections import Counter

from nlp_pipeline import StandardLibraryNLPPipeline
from utils import tokenize_words


LOGGER = logging.getLogger(__name__)


class QueryProcessor:
    """Prepare user claims and questions for searching and verification."""

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "because", "been", "being",
        "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
        "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it",
        "its", "me", "my", "of", "on", "or", "our", "she", "should", "that",
        "the", "their", "them", "there", "these", "they", "this", "those", "to",
        "was", "we", "were", "what", "when", "where", "which", "who", "why",
        "will", "with", "you", "your", "verify", "fact", "check", "true",
    }

    QUESTION_WORDS = {"what", "when", "where", "who", "why", "how", "does", "did", "is", "are", "can"}
    COUNTRIES = {
        "pakistan", "india", "china", "russia", "ukraine", "israel", "lebanon", "iran",
        "iraq", "syria", "palestine", "united states", "america", "usa", "france",
        "germany", "canada", "australia", "japan", "turkey", "beirut",
    }
    CITIES = {
        "beirut", "karachi", "lahore", "islamabad", "delhi", "mumbai", "new york",
        "washington", "london", "paris", "tokyo", "gaza", "jerusalem", "moscow",
    }
    CATEGORY_TERMS = {
        "Politics / International Affairs": {
            "trump", "president", "government", "minister", "election", "war", "attack",
            "israel", "beirut", "policy", "parliament", "senate", "court", "diplomat",
            "explosion", "port", "border", "military",
        },
        "Health": {"health", "vaccine", "covid", "virus", "disease", "doctor", "medicine", "hospital", "exercise", "cardiovascular"},
        "Science": {"science", "climate", "space", "study", "research", "experiment", "planet", "nasa", "earth", "moon", "moons", "satellite", "satellites", "mission", "earthquake", "magnitude", "aftershock", "seismic", "lightning", "storm", "weather", "thunder"},
        "Technology": {"technology", "ai", "software", "chip", "computer", "internet", "cyber"},
        "Sports": {"sports", "match", "football", "cricket", "goal", "tournament", "olympic", "final"},
        "Business": {"business", "market", "stock", "company", "bank", "economy", "price"},
        "Education": {"education", "school", "university", "student", "exam", "teacher", "degree"},
    }
    POSITIVE_WORDS = {"confirmed", "approved", "success", "true", "safe", "effective", "growth", "win"}
    NEGATIVE_WORDS = {"attack", "failed", "false", "war", "crisis", "denied", "killed", "risk", "fraud"}
    EVENT_WORDS = {
        "announce", "announced", "attack", "attacked", "declare", "declared", "discover",
        "discovered", "election", "forecast", "launch", "launched", "pandemic", "reduced",
        "report", "reported", "war", "conflict", "growth", "increase", "decrease",
        "earthquake", "struck", "strike", "killing", "killed", "injuring", "injured",
        "win", "won", "lose", "lost", "approve", "approved", "ban", "banned", "sign",
        "signed", "resign", "resigned", "appoint", "appointed", "release", "released",
        "publish", "published", "confirm", "confirmed", "deny", "denied", "open",
        "opened", "close", "closed", "merge", "merged", "acquire", "acquired",
    }
    UNIT_WORDS = {
        "percent", "percentage", "million", "billion", "trillion", "km", "kilometer",
        "kilometers", "mile", "miles", "meter", "meters", "kg", "tons", "year", "years",
        "month", "months", "day", "days", "moon", "moons", "satellite", "satellites",
        "magnitude", "province",
    }

    def __init__(self):
        self.pipeline = StandardLibraryNLPPipeline(stopwords=self.STOPWORDS)

    def process(self, text):
        """Return normalized query metadata as a dictionary."""
        original = (text or "").strip()
        pipeline_result = self.pipeline.process(original)
        tokens = pipeline_result["tokens"]
        keywords = self.extract_keywords(tokens)
        keyword_scores = self.rank_keywords(tokens)
        entities = pipeline_result["named_entities"]
        query_type = self.classify_question(original, tokens)
        intent = self.detect_intent(original, tokens)
        category = self.detect_category(tokens)
        complexity = self.estimate_complexity(original, tokens, entities)
        sentiment = self.sentiment(original)
        search_query = self.build_search_query(original, keywords, intent)
        claim_representation = self.claim_representation(original, tokens, entities, pipeline_result, intent, category)
        query_variants = self.generate_search_queries(original, keywords, claim_representation, intent)
        LOGGER.info("Processed query intent=%s type=%s keywords=%s", intent, query_type, keywords)
        return {
            "original": original,
            "normalized_text": pipeline_result["normalized_text"],
            "tokens": tokens,
            "keywords": keywords,
            "keyword_scores": keyword_scores,
            "entities": entities,
            "question_type": query_type,
            "intent": intent,
            "category": category,
            "complexity": complexity,
            "sentiment": sentiment,
            "nlp_pipeline": pipeline_result,
            "pipeline_summary": self.pipeline_summary(pipeline_result),
            "search_query": search_query,
            "query_variants": query_variants,
            "claim_representation": claim_representation,
        }

    def tokenize(self, text):
        """Tokenize and remove custom stopwords."""
        tokens = tokenize_words(text)
        return [token for token in tokens if token not in self.STOPWORDS and len(token) > 1]

    def extract_keywords(self, tokens, limit=8):
        """Extract high-value keywords and simple adjacent phrases."""
        counts = Counter(tokens)
        ranked = [word for word, _ in counts.most_common(limit)]
        bigrams = []
        for index in range(len(tokens) - 1):
            first, second = tokens[index], tokens[index + 1]
            if first != second and len(first) > 2 and len(second) > 2:
                bigrams.append(first + " " + second)
        for phrase, _ in Counter(bigrams).most_common(3):
            if phrase not in ranked:
                ranked.append(phrase)
        return ranked[:limit]

    def rank_keywords(self, tokens, limit=10):
        """Rank keywords with a transparent frequency and length heuristic."""
        counts = Counter(tokens)
        scored = []
        for token, count in counts.items():
            length_bonus = min(2.0, len(token) / 8.0)
            rarity_bonus = 1.0 if count == 1 else 0.4
            score = round((count * 2.0) + length_bonus + rarity_bonus, 2)
            scored.append({"keyword": token, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def extract_entities(self, text):
        """Detect people, organizations, countries, cities, and dates with regex heuristics."""
        original = text or ""
        people = []
        organizations = []
        dates = re.findall(
            r"\b(?:today|yesterday|tomorrow|(?:18|19|20)\d{2}|"
            r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
            original,
            flags=re.IGNORECASE,
        )
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", original)
        organization_markers = {"Organization", "University", "Agency", "Ministry", "Court", "Reuters", "BBC"}
        for item in capitalized:
            item_lower = item.lower()
            if item_lower in self.COUNTRIES or item_lower in self.CITIES:
                continue
            if any(marker in item for marker in organization_markers):
                organizations.append(item)
            else:
                people.append(item)
        acronyms = re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", original)
        organizations.extend(acronyms)
        lower = original.lower()
        countries = sorted({name.title() for name in self.COUNTRIES if re.search(r"\b" + re.escape(name) + r"\b", lower)})
        cities = sorted({name.title() for name in self.CITIES if re.search(r"\b" + re.escape(name) + r"\b", lower)})
        return {
            "people": self._unique(people),
            "organizations": self._unique(organizations),
            "countries": countries,
            "cities": cities,
            "dates": self._unique(dates),
        }

    def classify_question(self, text, tokens):
        """Classify the broad form of the user input."""
        lower = (text or "").lower().strip()
        first = tokenize_words(lower)[:1]
        if lower.endswith("?") or (first and first[0] in self.QUESTION_WORDS):
            if re.search(r"\b(18|19|20)\d{2}\b", lower):
                return "temporal_question"
            if "when" in lower:
                return "temporal_question"
            if "where" in lower:
                return "location_question"
            if "who" in lower:
                return "person_question"
            if "how many" in lower or "how much" in lower:
                return "quantity_question"
            return "general_question"
        if len(tokens) <= 12:
            return "short_claim_or_headline"
        return "long_claim"

    def detect_intent(self, text, tokens):
        """Detect whether the user is asking for news, research, or general fact checking."""
        lower = (text or "").lower()
        token_set = set(tokens)
        if {"headline", "news"} & token_set or any(mark in text for mark in ("BREAKING", "!!!")):
            return "news_verification"
        if {"study", "research", "paper", "journal", "article", "university", "doi"} & token_set:
            return "research_verification"
        if "fake" in token_set or "clickbait" in token_set:
            return "fake_news_detection"
        if lower.startswith(("is ", "are ", "was ", "were ", "did ", "does ")):
            return "claim_verification"
        return "general_fact_check"

    def detect_category(self, tokens):
        """Detect claim category from domain-specific vocabulary."""
        token_set = set(tokens)
        best_category = "General"
        best_score = 0
        for category, terms in self.CATEGORY_TERMS.items():
            score = len(token_set & terms)
            if score > best_score:
                best_category = category
                best_score = score
        return best_category

    def detect_claim_type(self, text, tokens, entities, intent, category):
        """Classify the semantic type of the claim for retrieval and explanation."""
        lower = (text or "").lower()
        token_set = set(tokens)
        types = []
        if entities.get("numbers") or re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", lower):
            types.append("numerical claim")
        if entities.get("dates") or re.search(r"\b(?:18|19|20)\d{2}\b", lower):
            types.append("historical claim")
        if entities.get("cities") or entities.get("countries") or {"city", "country", "capital", "located", "largest", "border"} & token_set:
            types.append("geographic claim")
        if category == "Science" or {"earth", "moon", "moons", "satellite", "planet", "nasa", "scientists", "study", "lightning", "storm", "weather"} & token_set:
            types.append("scientific claim")
        if {"earthquake", "aftershock", "flood", "storm", "cyclone", "wildfire"} & token_set:
            types.append("news claim")
        if any(mark in lower for mark in ("said", "says", "according to", "quote", "claimed")):
            types.append("quote/attribution claim")
        if category == "Politics / International Affairs":
            types.append("political claim")
        if intent == "news_verification" or {"news", "headline", "breaking", "reported"} & token_set:
            types.append("news claim")
        if {"report", "forecast", "bank", "institution", "agency", "ministry", "organization"} & token_set or entities.get("organizations"):
            types.append("institutional/report claim")
        if {"best", "worst", "beautiful", "should", "good", "bad", "better"} & token_set:
            types.append("opinion claim")
        if not types:
            types.append("general factual claim")
        if len(types) > 1:
            types.append("mixed claim")
        return types

    def claim_representation(self, original, tokens, entities, pipeline_result, intent, category):
        """Build a structured claim object used by retrieval, stance, and UI layers."""
        structure = pipeline_result.get("claim_structure", {})
        numbers = entities.get("numbers", [])
        units = self.extract_units(tokens)
        events = [token for token in tokens if token in self.EVENT_WORDS]
        claim_types = self.detect_claim_type(original, tokens, entities, intent, category)
        normalized_text = pipeline_result.get("normalized_text", original)
        return {
            "subject": structure.get("subject", ""),
            "relation": structure.get("predicate", ""),
            "predicate": structure.get("predicate", ""),
            "object": structure.get("object", ""),
            "normalized_text": normalized_text,
            "entities": entities,
            "people": entities.get("people", []),
            "organizations": entities.get("organizations", []),
            "locations": entities.get("countries", []) + entities.get("cities", []),
            "dates": entities.get("dates", []),
            "events": self._unique(events),
            "numbers": numbers,
            "units": units,
            "keywords": tokens[:10],
            "claim_type": ", ".join(claim_types),
            "claim_types": claim_types,
            "category": category,
        }

    def extract_units(self, tokens):
        """Extract simple measurement/count units near numeric or scientific claims."""
        units = []
        for token in tokens:
            if token in self.UNIT_WORDS:
                units.append(token)
        return self._unique(units)

    def generate_search_queries(self, original, keywords, representation, intent):
        """Generate a broad open-domain query bundle for live verification."""
        normalized = representation.get("normalized_text", original)
        subject = representation.get("subject", "").strip()
        relation = representation.get("relation", "").strip()
        obj = representation.get("object", "").strip()
        entities = []
        for group in ("people", "organizations", "locations"):
            entities.extend(representation.get(group, []))
        compact_keywords = " ".join([keyword for keyword in keywords if " " not in keyword][:8]).strip()
        entity_focus = " ".join(self._unique(entities[:5])).strip()
        relation_query = " ".join(part for part in (subject, relation, obj) if part).strip()
        core_query = relation_query or entity_focus or compact_keywords or normalized
        claim_type = representation.get("claim_type", "fact")
        queries = [
            {"type": "exact_claim", "query": original},
            {"type": "simplified_claim", "query": compact_keywords or normalized},
            {"type": "entity_relation", "query": relation_query or entity_focus or original},
            {"type": "verification_true", "query": "is " + core_query + " true"},
            {"type": "verification_false", "query": "is " + core_query + " false"},
            {"type": "fact_check", "query": core_query + " fact check"},
            {"type": "official_source", "query": (entity_focus or core_query) + " official source"},
            {"type": "credible_source", "query": (entity_focus or core_query) + " credible source"},
            {"type": "contradiction", "query": core_query + " false debunked hoax"},
            {"type": "evidence", "query": core_query + " evidence"},
            {"type": "paraphrased", "query": self._paraphrase_query(core_query, " ".join(representation.get("numbers", []) + representation.get("units", [])), claim_type)},
        ]
        queries.extend(self._event_queries(original, representation, intent))
        if intent == "research_verification" or "scientific" in claim_type:
            queries.append({"type": "research_source", "query": core_query + " research scientific source"})
        if "institutional" in claim_type:
            queries.append({"type": "official_report", "query": core_query + " official report"})
        return self._dedupe_queries(queries)

    def _dedupe_queries(self, queries):
        seen = set()
        output = []
        for item in queries:
            query = " ".join((item.get("query") or "").split())
            key = query.lower()
            if query and key not in seen:
                seen.add(key)
                output.append({"type": item["type"], "query": query})
        return output

    def _event_queries(self, original, representation, intent):
        tokens = tokenize_words(original)
        token_set = set(tokens)
        event_terms = representation.get("events", []) or [
            token for token in tokens if token in self.EVENT_WORDS
        ]
        claim_type = representation.get("claim_type", "")
        should_use_news = (
            bool(event_terms)
            or "news" in claim_type
            or intent == "news_verification"
            or bool({"today", "yesterday", "reported", "according"} & token_set)
        )
        if not should_use_news:
            return []
        locations = " ".join(representation.get("locations", [])[:3])
        organizations = " ".join(representation.get("organizations", [])[:3])
        people = " ".join(representation.get("people", [])[:3])
        numbers = " ".join(representation.get("numbers", [])[:3])
        decisive_terms = [
            token for token in tokens
            if token not in self.STOPWORDS and (
                token in event_terms
                or token in self.NEGATIVE_WORDS
                or token in self.POSITIVE_WORDS
                or token in {"dead", "injured", "casualties", "official", "state", "media", "forecast", "report"}
            )
        ]
        query_core = " ".join(
            part for part in [
                people,
                organizations,
                locations,
                " ".join(event_terms[:4]),
                numbers,
                " ".join(decisive_terms[:5]),
            ]
            if part
        ).strip()
        if not query_core:
            query_core = " ".join([token for token in tokens if token not in self.STOPWORDS][:8])
        if not query_core:
            return []
        queries = [
            {"type": "event_news", "query": query_core + " news"},
            {"type": "official_event", "query": query_core + " official source"},
            {"type": "major_news", "query": query_core + " Reuters AP BBC"},
        ]
        if "institutional" in claim_type:
            queries.insert(1, {"type": "institutional_news", "query": query_core + " official report"})
        return queries

    def _paraphrase_query(self, relation_query, numbers, claim_type):
        base = relation_query or ""
        if numbers:
            return base + " actual number " + numbers
        if "geographic" in claim_type:
            return base + " location country official"
        if "scientific" in claim_type:
            return base + " scientific explanation"
        return base + " evidence"

    def estimate_complexity(self, text, tokens, entities):
        """Estimate query complexity from length, entities, and dates."""
        entity_count = sum(len(values) for values in entities.values())
        score = len(tokens) + entity_count + len(re.findall(r"\b(18|19|20)\d{2}\b", text or ""))
        if score >= 16:
            return "High"
        if score >= 7:
            return "Medium"
        return "Low"

    def sentiment(self, text):
        """Rule-based sentiment and tone estimate."""
        tokens = set(tokenize_words(text))
        positive = len(tokens & self.POSITIVE_WORDS)
        negative = len(tokens & self.NEGATIVE_WORDS)
        if positive > negative:
            label = "Positive"
        elif negative > positive:
            label = "Negative"
        else:
            label = "Neutral"
        return {"label": label, "positive_terms": positive, "negative_terms": negative}

    def build_search_query(self, original, keywords, intent):
        """Build a compact web-search query."""
        single_terms = []
        for keyword in keywords:
            if " " not in keyword:
                single_terms.append(keyword)
        if single_terms:
            query = " ".join(single_terms[:7])
        else:
            query = original
        if intent == "research_verification":
            query += " research evidence"
        elif intent == "news_verification":
            query += " news fact check"
        else:
            query += " reliable source"
        return query.strip()

    def pipeline_summary(self, pipeline_result):
        """Return a compact teaching/demo summary of the NLP pipeline."""
        return {
            "stages": pipeline_result.get("stages", []),
            "normalization": pipeline_result.get("normalized_text", ""),
            "sentence_count": pipeline_result.get("sentence_count", 0),
            "token_count_after_stopword_removal": pipeline_result.get("token_count", 0),
            "top_terms": Counter(pipeline_result.get("tokens", [])).most_common(5),
            "lemmas": pipeline_result.get("lemmas", [])[:10],
            "bigrams": pipeline_result.get("ngrams", {}).get("bigrams", [])[:5],
            "pos_sample": pipeline_result.get("pos_tags", [])[:10],
            "entities": pipeline_result.get("named_entities", {}),
            "noun_phrases": pipeline_result.get("noun_phrases", [])[:5],
            "verb_phrases": pipeline_result.get("verb_phrases", [])[:5],
            "claim_structure": pipeline_result.get("claim_structure", {}),
            "lexical_diversity": pipeline_result.get("lexical_diversity", 0),
            "lexical_density": pipeline_result.get("lexical_density", 0),
        }

    def _unique(self, values):
        seen = set()
        result = []
        for value in values:
            cleaned = (value or "").strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result
