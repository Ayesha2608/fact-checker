"""Professional claim verification and evidence analysis engine."""

import logging
import queue
import hashlib
import threading
from collections import Counter

from confidence import ConfidenceEngine
from contradiction_detector import ContradictionDetector
from credibility import CredibilityAnalyzer
from database import DatabaseManager
from evidence_extractor import EvidenceExtractor
from hallucination_detector import HallucinationDetector
from query_processor import QueryProcessor
from search_engine import SearchEngine
from utils import clean_text, display_timestamp, domain_from_url, format_claim_id, tokenize_words
from web_scraper import WebScraper


LOGGER = logging.getLogger(__name__)


class FactCheckingEngine:
    """Coordinate retrieval, NLP analysis, explainability, metrics, and storage."""

    def __init__(self, database_path="fact_checker.db", max_sources=12):
        self.query_processor = QueryProcessor()
        self.search_engine = SearchEngine(timeout=5, max_results=max_sources)
        self.scraper = WebScraper()
        self.credibility = CredibilityAnalyzer()
        self.extractor = EvidenceExtractor()
        self.detector = ContradictionDetector()
        self.confidence = ConfidenceEngine()
        self.hallucination = HallucinationDetector()
        self.database = DatabaseManager(database_path)
        self.max_sources = max_sources

    def verify_claim(self, claim, mode="general"):
        """Verify a claim and return a complete platform-grade result dictionary."""
        timestamp = display_timestamp()
        processed = self.query_processor.process(claim)
        search_bundle = self._search_with_fallbacks(claim, processed)
        results = search_bundle["results"]
        pages = self._fetch_pages_threaded(results)
        evidence_items, sources, scores = self._analyze_pages(claim, processed, pages)
        source_clusters = self._source_clusters(sources, evidence_items)
        retrieval_analysis = self._retrieval_analysis(results, pages, search_bundle["diagnostics"])
        retrieval_analysis["independent_source_count"] = len(source_clusters)
        retrieval_analysis["duplicate_source_clusters"] = len([cluster for cluster in source_clusters if cluster["source_count"] > 1])
        reliability_analysis = self._reliability_analysis(sources, source_clusters)
        quality_metrics = self._quality_metrics(claim, processed, evidence_items, sources, retrieval_analysis, reliability_analysis, source_clusters)
        risk_analysis = self._risk_analysis(claim, processed, sources, quality_metrics, mode)
        status = self._status(evidence_items, retrieval_analysis)
        fake_news_report = self.analyze_fake_news(claim)
        unsupported_claim_report = self._unsupported_claim_assessment(claim, processed, retrieval_analysis, fake_news_report)
        search_diagnostics = retrieval_analysis.get("search_diagnostics", {})
        search_reached = self._search_reached(search_diagnostics)
        result_count = search_diagnostics.get("results_returned", retrieval_analysis.get("sources_searched", 0))
        readable_pages = retrieval_analysis.get("sources_successfully_retrieved", 0)
        verdict_rule = ""

        if status == "INSUFFICIENT DATA":
            verdict = "Insufficient Evidence"
            confidence = {"percentage": None, "level": "Not Available"}
            explanation = self._insufficient_data_explanation(retrieval_analysis)
            verdict_rule = "technical_retrieval_failure"
        elif status == "UNCERTAIN":
            if unsupported_claim_report["label"]:
                verdict = unsupported_claim_report["label"]
                confidence = unsupported_claim_report["confidence"]
                explanation = self._likely_fake_claim_explanation(retrieval_analysis, unsupported_claim_report)
                verdict_rule = "fabricated_claim_pattern"
            elif search_reached and result_count == 0:
                verdict = "Unsupported Claim"
                confidence = {"percentage": 0, "level": "Very Low"}
                explanation = self._unsupported_claim_explanation(retrieval_analysis)
                verdict_rule = "search_worked_no_support"
            elif search_reached and result_count > 0 and readable_pages == 0:
                verdict = "Insufficient Evidence"
                confidence = {"percentage": None, "level": "Not Available"}
                explanation = self._insufficient_data_explanation(retrieval_analysis)
                verdict_rule = "search_results_unreadable"
            elif search_reached:
                verdict = "Unsupported Claim"
                confidence = {"percentage": 0, "level": "Very Low"}
                explanation = self._unsupported_claim_explanation(retrieval_analysis)
                verdict_rule = "search_worked_no_support"
            else:
                verdict = "Insufficient Evidence"
                confidence = {"percentage": None, "level": "Not Available"}
                explanation = self._insufficient_data_explanation(retrieval_analysis)
                verdict_rule = "technical_retrieval_failure"
        else:
            confidence = self._confidence(scores, evidence_items, quality_metrics)
            verdict = self._determine_verdict(scores, evidence_items, confidence, quality_metrics)
            explanation = self._explain(verdict, confidence, scores, evidence_items, quality_metrics, reliability_analysis, source_clusters)
            verdict_rule = self._verdict_rule_from_verdict(verdict)

        citations = self._citations(sources)
        answer = self._build_answer(status, verdict, confidence, explanation, evidence_items)
        hallucination_report = self.hallucination.analyze(answer, evidence_items) if evidence_items else {
            "unsupported_statements": [],
            "warning": False,
            "message": "No answer claims were generated beyond retrieval diagnostics",
        }
        result = {
            "claim_id": "",
            "timestamp": timestamp,
            "query": claim,
            "processed_query": processed,
            "status": status,
            "verdict": verdict,
            "confidence": confidence,
            "supporting_sources": len([item for item in evidence_items if item["stance"] == "support"]),
            "contradicting_sources": len([item for item in evidence_items if item["stance"] == "contradict"]),
            "neutral_sources": len([item for item in evidence_items if item["stance"] == "neutral"]),
            "retrieval_analysis": retrieval_analysis,
            "source_reliability_analysis": reliability_analysis,
            "quality_metrics": quality_metrics,
            "risk_analysis": risk_analysis,
            "evidence": evidence_items,
            "sources": sources,
            "source_clusters": source_clusters,
            "citations": citations,
            "verification_diagnostics": self._verification_diagnostics(
                claim,
                processed,
                search_bundle,
                retrieval_analysis,
                reliability_analysis,
                quality_metrics,
                evidence_items,
                verdict_rule,
            ),
            "final_verdict_rule": verdict_rule,
            "explanation": explanation,
            "answer": answer,
            "hallucination_report": hallucination_report,
            "fake_news_report": fake_news_report,
            "unsupported_claim_report": unsupported_claim_report,
        }
        fact_check_id = self.database.save_fact_check(result)
        result["fact_check_id"] = fact_check_id
        result["claim_id"] = format_claim_id(fact_check_id)
        LOGGER.info("Verified claim status=%s verdict=%s confidence=%s", status, verdict, confidence)
        return result

    def _search_with_fallbacks(self, claim, processed):
        """Try multiple query forms before declaring search failure."""
        query_variants = processed.get("query_variants", [])
        candidates = [item.get("query", "") for item in query_variants]
        candidates.extend(
            [
                processed.get("search_query", ""),
                processed.get("original", "") or claim,
                " ".join([keyword for keyword in processed.get("keywords", []) if " " not in keyword][:6]),
            ]
        )
        original = processed.get("original", "") or claim
        compact_keywords = " ".join([keyword for keyword in processed.get("keywords", []) if " " not in keyword][:8])
        candidates.extend(
            [
                original + " fact check",
                original + " reliable source",
                compact_keywords + " fact check",
                compact_keywords + " evidence",
            ]
        )
        seen = set()
        attempts = []
        best_bundle = None
        best_score = -1
        merged_results = []
        seen_urls = set()
        for query in candidates:
            query = (query or "").strip()
            if not query or query.lower() in seen:
                continue
            seen.add(query.lower())
            bundle = self.search_engine.search_with_diagnostics(query, self.max_sources)
            diagnostics = bundle.get("diagnostics", {})
            result_count = len(bundle.get("results", []))
            for result in bundle.get("results", []):
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    result["query_type"] = self._query_variant_type(query, query_variants)
                    result["query_used"] = query
                    merged_results.append(result)
            attempts.append(
                {
                    "query": query,
                    "query_type": self._query_variant_type(query, query_variants),
                    "status_code": diagnostics.get("status_code"),
                    "results_returned": result_count,
                    "error": diagnostics.get("error", ""),
                }
            )
            domains = {result.get("domain", "") for result in bundle.get("results", []) if result.get("domain")}
            score = (result_count * 3) + len(domains)
            if self._query_contains_official_terms(query):
                score += 2
            if score > best_score:
                best_score = score
                best_bundle = bundle
            if len(attempts) >= 5 and len(merged_results) >= self.max_sources:
                break
        if best_bundle is None:
            best_bundle = {"results": [], "diagnostics": {"query": claim, "error": "No search attempts were executed"}}
        if merged_results:
            best_bundle["results"] = self._rank_search_results(merged_results, processed)[: self.max_sources]
            best_bundle["diagnostics"]["results_returned"] = len(best_bundle["results"])
        best_bundle["diagnostics"]["fallback_attempts"] = attempts
        return best_bundle

    def _rank_search_results(self, results, processed):
        """Prefer result metadata that matches the claim semantics, not generic pages."""
        claim_tokens = set(processed.get("tokens", []))
        representation = processed.get("claim_representation", {})
        entity_terms = set()
        for group in ("people", "organizations", "locations"):
            for value in representation.get(group, []):
                entity_terms.update(tokenize_words(value))
        event_terms = set(representation.get("events", []))
        number_terms = set(representation.get("numbers", []))
        claim_type = representation.get("claim_type", "")
        high_value_domains = (
            ".gov", ".edu", ".ac.", "reuters.com", "apnews.com", "bbc.", "who.int",
            "un.org", "nasa.gov", "britannica.com", "nature.com", "science.org",
        )
        low_value_markers = (
            "dictionary", "definition", "meaning", "pronunciation", "translation",
            "translate", "thesaurus", "grammar", "wordreference", "wiktionary",
        )

        def score(result):
            text = " ".join([result.get("title", ""), result.get("snippet", ""), result.get("domain", "")]).lower()
            tokens = set(tokenize_words(text))
            domain = result.get("domain", "")
            value = 0
            value += len(tokens & claim_tokens) * 4
            value += len(tokens & entity_terms) * 8
            value += len(tokens & event_terms) * 7
            value += len(tokens & number_terms) * 8
            if result.get("snippet"):
                value += 8
            if result.get("query_type") in {"event_news", "major_news", "official_event", "institutional_news"}:
                value += 10
            if any(marker in domain for marker in high_value_domains):
                value += 8
            if "news" in claim_type and any(source in domain for source in ("reuters.com", "apnews.com", "bbc.", "thehindu.com", "straitstimes.com")):
                value += 7
            if "definition" not in claim_type and any(marker in text for marker in low_value_markers):
                value -= 30
            if domain in {"bing.com", "duckduckgo.com"}:
                value -= 15
            return value

        ranked = sorted(results, key=score, reverse=True)
        return [result for result in ranked if score(result) > -10]

    def _query_variant_type(self, query, variants):
        for variant in variants:
            if (variant.get("query") or "").strip().lower() == (query or "").strip().lower():
                return variant.get("type", "fallback")
        return "fallback"

    def _query_contains_official_terms(self, query):
        lower = (query or "").lower()
        return "official" in lower or "reliable" in lower or "fact check" in lower

    def analyze_fake_news(self, headline):
        """Estimate fake-news probability from transparent text rules."""
        text = headline or ""
        lower = text.lower()
        score = 0
        reasons = []
        if sum(1 for char in text if char.isupper()) > max(8, len(text) * 0.35):
            score += 25
            reasons.append("Excessive capitalization")
        if text.count("!") >= 2:
            score += 20
            reasons.append("Too many exclamation marks")
        clickbait = ["shocking", "you won't believe", "secret", "exposed", "miracle", "must see", "breaking"]
        hits = [phrase for phrase in clickbait if phrase in lower]
        if hits:
            score += 15 + (len(hits) * 5)
            reasons.append("Clickbait phrase(s): " + ", ".join(hits))
        if "source" not in lower and "according to" not in lower:
            score += 15
            reasons.append("No visible source attribution")
        probability = min(100, score)
        if probability >= 70:
            level = "High"
        elif probability >= 40:
            level = "Medium"
        else:
            level = "Low"
        return {"probability": probability, "level": level, "reasons": reasons}

    def _unsupported_claim_assessment(self, claim, processed, retrieval, fake_news_report=None):
        """Estimate whether an unsupported live-search claim looks fabricated."""
        fake_news_report = fake_news_report or self.analyze_fake_news(claim)
        diagnostics = retrieval.get("search_diagnostics", {})
        fallback_attempts = diagnostics.get("fallback_attempts", [])
        attempt_results = [attempt.get("results_returned", 0) for attempt in fallback_attempts]
        attempt_statuses = [attempt.get("status_code") for attempt in fallback_attempts]
        search_reached = any(status == 200 for status in attempt_statuses)
        if not search_reached:
            search_reached = any(
                diagnostics.get(key) == 200
                for key in ("status_code", "fallback_status_code", "secondary_status_code")
            )
        if not search_reached:
            return {
                "label": "",
                "score": 0,
                "confidence": {"percentage": 0, "level": "Very Low"},
                "reasons": [],
                "triggered": False,
            }

        tokens = set(processed.get("tokens", []))
        lower = (claim or "").lower()
        score = min(35, fake_news_report.get("probability", 0))
        reasons = []
        if fake_news_report.get("reasons"):
            reasons.extend(fake_news_report.get("reasons"))
        if fallback_attempts and all(results == 0 for results in attempt_results):
            score += 18
            reasons.append("Multiple live search attempts returned no corroborating results")
        if retrieval.get("sources_searched", 0) == 0 and retrieval.get("pages_retrieved", 0) == 0:
            score += 12
            reasons.append("No live source could be matched closely enough to the claim")
        extraordinary_terms = {
            "flying", "alien", "aliens", "zombie", "zombies", "teleport", "teleportation",
            "miracle", "immortal", "immortality", "monster", "dragons", "dragon",
            "ghost", "ghosts", "magic", "supernatural",
        }
        sensational_terms = {"shocking", "secret", "exposed", "breaking", "viral", "unbelievable"}
        has_extraordinary_terms = bool(tokens & extraordinary_terms)
        has_sensational_terms = bool(tokens & sensational_terms)
        has_discovery_pattern = bool({"scientist", "scientists", "researcher", "researchers"} & tokens and {"discover", "discovered"} & tokens)
        if has_extraordinary_terms:
            score += 22
            reasons.append("The claim uses extraordinary or implausible event language")
        if has_sensational_terms:
            score += 10
            reasons.append("The wording resembles sensational headline framing")
        if has_discovery_pattern:
            score += 12
            reasons.append("The claim presents a discovery headline but no credible corroboration was found")
        if (has_extraordinary_terms or has_discovery_pattern) and retrieval.get("sources_successfully_retrieved", 0) > 0:
            score += 12
            reasons.append("Related live pages were retrieved, but none verified the exact event described in the claim")
        has_source_attribution = any("source attribution" in reason.lower() for reason in reasons)
        if (has_extraordinary_terms or has_sensational_terms or has_discovery_pattern) and not has_source_attribution and "according to" not in lower and "source" not in lower and "http://" not in lower and "https://" not in lower:
            score += 8
            reasons.append("The claim contains no direct source attribution")
        if (has_extraordinary_terms or has_discovery_pattern) and any(processed.get("entities", {}).values()):
            score += 5
            reasons.append("Named entities are present, but live search did not verify the event around them")

        score = min(100, score)
        if score >= 75:
            label = "Fake Claim"
            confidence = {"percentage": 72, "level": "High"}
        elif score >= 55:
            label = "Likely Fake Claim"
            confidence = {"percentage": 61, "level": "Medium"}
        else:
            label = ""
            confidence = {"percentage": 0, "level": "Very Low"}

        return {
            "label": label,
            "score": score,
            "confidence": confidence,
            "reasons": self._unique(reasons),
            "triggered": bool(label),
        }

    def rank_research_sources(self, sources):
        """Return sources sorted for research article verification."""
        return sorted(
            sources,
            key=lambda source: (source.get("reliability", 0), source.get("domain", "").endswith(".edu")),
            reverse=True,
        )

    def _fetch_pages_threaded(self, search_results):
        output = queue.Queue()
        threads = []

        def worker(result):
            page = self.scraper.fetch(result.get("url", ""))
            output.put((result, page))

        for result in search_results[: self.max_sources]:
            thread = threading.Thread(target=worker, args=(result,))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        pages = []
        while not output.empty():
            pages.append(output.get())
        return pages

    def _analyze_pages(self, claim, processed, pages):
        evidence_items = []
        sources = []
        scores = {"support": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for search_result, page in pages:
            url = search_result.get("url", "")
            credibility = self.credibility.score(url)
            title = page.get("title") or search_result.get("title", "")
            extracted = []
            if page.get("accepted"):
                extracted = self.extractor.extract(
                    page.get("text", ""),
                    processed["keywords"],
                    claim,
                    claim_representation=processed.get("claim_representation", {}),
                )
            fallback_text = ". ".join(
                part for part in [
                    search_result.get("title", ""),
                    search_result.get("snippet", ""),
                ]
                if part
            )
            if fallback_text:
                fallback_extracted = self.extractor.extract(
                    fallback_text,
                    processed["keywords"],
                    claim,
                    max_sentences=3,
                    claim_representation=processed.get("claim_representation", {}),
                )
                for item in fallback_extracted:
                    item["evidence_origin"] = "search_result_metadata"
                extracted = self._merge_candidate_evidence(extracted, fallback_extracted)
            source = self._source_record(url, title, credibility, "neutral", "", page)
            if not extracted:
                sources.append(source)
                continue
            source_candidates = []
            for candidate in extracted[:3]:
                if not self._candidate_relevant(candidate, processed):
                    continue
                stance_text = (candidate["sentence"] + " " + candidate.get("context", "")).strip()
                stance = self.detector.classify(claim, stance_text, processed.get("claim_representation", {}))
                similarity_factor = 1 + (candidate.get("combined_similarity", 0) / 100) * 0.25
                semantic_factor = 1 + (candidate.get("semantic_score", 0) / 100) * 0.35
                weighted = candidate["relevance"] * credibility["reliability"] * similarity_factor * semantic_factor
                if stance == "support":
                    scores["support"] += weighted
                elif stance == "contradict":
                    scores["contradiction"] += weighted
                else:
                    scores["neutral"] += weighted * 0.5
                evidence_record = {
                    "url": url,
                    "title": title,
                    "domain": credibility["domain"],
                    "reliability": credibility["reliability"],
                    "credibility_reason": credibility["reason"],
                    "stance": stance,
                    "sentence": candidate["sentence"],
                    "context": candidate.get("context", ""),
                    "relevance": candidate["relevance"],
                    "claim_coverage": candidate.get("claim_coverage", 0),
                    "jaccard_similarity": candidate.get("jaccard_similarity", 0),
                    "cosine_similarity": candidate.get("cosine_similarity", 0),
                    "sequence_similarity": candidate.get("sequence_similarity", 0),
                    "combined_similarity": candidate.get("combined_similarity", 0),
                    "semantic_score": candidate.get("semantic_score", 0),
                    "entity_overlap": candidate.get("entity_overlap", 0),
                    "subject_match": candidate.get("subject_match", False),
                    "predicate_match": candidate.get("predicate_match", False),
                    "object_match": candidate.get("object_match", False),
                    "number_match": candidate.get("number_match", False),
                    "date_match": candidate.get("date_match", False),
                    "weighted_score": round(weighted, 4),
                    "evidence_score": min(100, int(round((candidate.get("combined_similarity", 0) * 0.45) + (candidate.get("semantic_score", 0) * 0.35) + (candidate.get("claim_coverage", 0) * 0.2)))),
                    "text_length": page.get("text_length", 0),
                }
                evidence_items.append(evidence_record)
                source_candidates.append(evidence_record)
            if source_candidates:
                best = max(source_candidates, key=lambda item: item["weighted_score"])
                source.update({"stance": best["stance"], "evidence": best["sentence"]})
            sources.append(source)
        return evidence_items, sources, scores

    def _merge_candidate_evidence(self, primary, fallback):
        seen = {clean_text(item.get("sentence", "")).lower() for item in primary}
        merged = list(primary)
        for item in fallback:
            key = clean_text(item.get("sentence", "")).lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        merged.sort(key=lambda item: (item.get("semantic_score", 0), item.get("relevance", 0)), reverse=True)
        return merged[:6]

    def _candidate_relevant(self, candidate, processed):
        """Reject background text that misses required claim anchors."""
        text = (candidate.get("sentence", "") + " " + candidate.get("context", "")).lower()
        coverage = candidate.get("claim_coverage", 0)
        sentence_tokens = set(tokenize_words(text))
        if self._topic_mismatch(processed, sentence_tokens):
            return False
        claim_segments = self._claim_segments(processed)
        if not any(claim_segments.values()):
            return coverage >= 45
        segment_hits = {name: bool(sentence_tokens & terms) for name, terms in claim_segments.items() if terms}
        rare_terms = self._claim_rare_terms(processed)
        entity_terms = self._claim_entity_terms(processed)
        entity_overlap = len(sentence_tokens & entity_terms)
        rare_overlap = len(sentence_tokens & rare_terms)
        is_long_or_news = processed.get("question_type") == "long_claim" or "news" in processed.get("claim_representation", {}).get("claim_type", "")
        if not all(segment_hits.values()):
            if is_long_or_news and coverage >= 35 and (entity_overlap >= 1 or rare_overlap >= 2):
                return True
            return False
        if coverage < 25:
            return False
        if all(segment_hits.values()) and coverage >= 60:
            return True
        if entity_overlap >= 1 or rare_overlap >= 1:
            return True
        return False

    def _topic_mismatch(self, processed, sentence_tokens):
        """Reject obvious off-topic metaphors for scientific/natural claims."""
        claim_type = processed.get("claim_representation", {}).get("claim_type", "")
        if "scientific" not in claim_type:
            return False
        science_terms = {
            "science", "scientific", "research", "study", "climate", "space", "planet",
            "nasa", "earth", "moon", "moons", "satellite", "satellites", "earthquake",
            "magnitude", "seismic", "lightning", "storm", "weather", "thunder",
            "strike", "strikes", "struck", "bolt", "bolts", "natural",
        }
        off_topic_terms = {
            "nba", "warriors", "team", "coach", "player", "sports", "racing", "racecar",
            "movie", "film", "character", "game", "monster", "cathedral", "souls",
            "season", "champion", "dynasty", "piston", "cup",
        }
        if not (sentence_tokens & off_topic_terms):
            return False
        return len(sentence_tokens & science_terms) <= 2

    def _source_clusters(self, sources, evidence_items):
        """Group duplicate or near-duplicate sources so confidence uses independence."""
        clusters = {}
        evidence_by_url = {}
        for item in evidence_items:
            evidence_by_url.setdefault(item.get("url", ""), []).append(item)
        for source in sources:
            domain = source.get("domain", "")
            evidence_text = source.get("evidence", "")
            fingerprint = self._content_fingerprint(evidence_text or source.get("title", ""))
            key = fingerprint or domain
            if not key:
                key = source.get("url", "")
            cluster = clusters.setdefault(
                key,
                {
                    "cluster_id": "cluster-" + hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:8],
                    "representative_domain": domain,
                    "source_count": 0,
                    "urls": [],
                    "stance_counts": {"support": 0, "contradict": 0, "neutral": 0},
                    "max_reliability": 0,
                    "best_evidence_score": 0,
                },
            )
            cluster["source_count"] += 1
            if source.get("url"):
                cluster["urls"].append(source.get("url"))
            cluster["stance_counts"][source.get("stance", "neutral")] = cluster["stance_counts"].get(source.get("stance", "neutral"), 0) + 1
            cluster["max_reliability"] = max(cluster["max_reliability"], source.get("reliability", 0))
            for evidence in evidence_by_url.get(source.get("url", ""), []):
                evidence["source_cluster_id"] = cluster["cluster_id"]
                cluster["best_evidence_score"] = max(cluster["best_evidence_score"], evidence.get("evidence_score", 0))
        return sorted(clusters.values(), key=lambda item: (item["max_reliability"], item["best_evidence_score"]), reverse=True)

    def _content_fingerprint(self, text):
        tokens = [token for token in tokenize_words(text) if len(token) > 3]
        if not tokens:
            return ""
        return hashlib.sha1(" ".join(tokens[:24]).encode("utf-8", errors="ignore")).hexdigest()[:10]

    def _retrieval_analysis(self, results, pages, search_diagnostics):
        page_diagnostics = []
        retrieved_domains = []
        contacted_domains = []
        for search_result, page in pages:
            domain = domain_from_url(search_result.get("url", ""))
            contacted_domains.append(domain)
            if page.get("retrieved"):
                retrieved_domains.append(domain)
            page_diagnostics.append(
                {
                    "url": search_result.get("url", ""),
                    "domain": domain,
                    "status_code": page.get("status_code"),
                    "retrieved": bool(page.get("retrieved")),
                    "accepted": bool(page.get("accepted")),
                    "text_length": page.get("text_length", 0),
                    "response_bytes": page.get("response_bytes", 0),
                    "extraction_quality": page.get("extraction_quality", 0),
                    "rejection_reason": page.get("rejection_reason", ""),
                    "error": page.get("error", ""),
                }
            )
        successful = [page for _, page in pages if page.get("accepted")]
        failed = [page for _, page in pages if not page.get("accepted")]
        average_quality = 0
        if pages:
            average_quality = int(round(sum(page.get("extraction_quality", 0) for _, page in pages) / len(pages)))
        return {
            "search_diagnostics": search_diagnostics,
            "sources_searched": len(results),
            "sources_successfully_retrieved": len(successful),
            "failed_sources": len(failed),
            "domains_contacted": sorted({domain for domain in contacted_domains if domain}),
            "domains_retrieved": sorted({domain for domain in retrieved_domains if domain}),
            "average_retrieval_quality": average_quality,
            "pages_retrieved": len([page for _, page in pages if page.get("retrieved")]),
            "pages_rejected": len(failed),
            "extraction_success_rate": int(round((len(successful) / max(1, len(pages))) * 100)),
            "page_diagnostics": page_diagnostics,
        }

    def _reliability_analysis(self, sources, source_clusters=None):
        counts = Counter()
        scores = []
        tiers = Counter()
        for source in sources:
            reason = source.get("reliability_reason", "")
            domain = source.get("domain", "")
            scores.append(source.get("reliability", 0))
            tiers[source.get("credibility_tier", "Tier 3")] += 1
            if "Government" in reason:
                counts["government"] += 1
            elif "Educational" in reason:
                counts["educational"] += 1
            elif "Research" in reason or "scientific" in reason:
                counts["research"] += 1
            elif "news" in reason.lower() or "fact-checking" in reason:
                counts["news"] += 1
            elif domain:
                counts["other"] += 1
        weighted = int(round((sum(scores) / max(1, len(scores))) * 10))
        independent_scores = [cluster.get("max_reliability", 0) for cluster in (source_clusters or [])]
        independent_weighted = int(round((sum(independent_scores) / max(1, len(independent_scores))) * 10))
        return {
            "government_sources": counts["government"],
            "educational_sources": counts["educational"],
            "research_sources": counts["research"],
            "news_sources": counts["news"],
            "other_sources": counts["other"],
            "weighted_reliability_score": weighted,
            "independent_weighted_reliability_score": independent_weighted,
            "tier_distribution": dict(tiers),
            "independent_source_count": len(source_clusters or []),
        }

    def _quality_metrics(self, claim, processed, evidence_items, sources, retrieval, reliability, source_clusters=None):
        claim_terms = set(processed.get("tokens", []))
        evidence_terms = set()
        for item in evidence_items:
            evidence_terms.update(tokenize_words(item.get("sentence", "")))
        coverage = int(round((len(claim_terms & evidence_terms) / max(1, len(claim_terms))) * 100))
        strength = 0
        if evidence_items:
            strength = int(round(sum(item.get("evidence_score", item["reliability"] * item["relevance"]) for item in evidence_items) / len(evidence_items)))
            strength = min(100, strength)
        unique_domains = len({source.get("domain") for source in sources if source.get("domain")})
        diversity = int(round((unique_domains / max(1, len(sources))) * 100))
        reliability_score = reliability.get("independent_weighted_reliability_score", reliability.get("weighted_reliability_score", 0))
        retrieval_quality = retrieval.get("average_retrieval_quality", 0)
        support = len([item for item in evidence_items if item.get("stance") == "support"])
        contradiction = len([item for item in evidence_items if item.get("stance") == "contradict"])
        neutral = len([item for item in evidence_items if item.get("stance") == "neutral"])
        decisive = max(support, contradiction)
        agreement = int(round((decisive / max(1, support + contradiction + neutral)) * 100)) if evidence_items else 0
        duplicate_penalty = max(0, len(sources) - len(source_clusters or [])) * 3
        confidence_inputs = {
            "evidence_strength": max(0, strength - duplicate_penalty),
            "source_credibility": reliability_score,
            "agreement": agreement,
            "coverage": coverage,
        }
        overall = self.confidence.calculate_breakdown(**confidence_inputs)["percentage"]
        return {
            "coverage_score": coverage,
            "evidence_strength": strength,
            "evidence_strength_adjusted": confidence_inputs["evidence_strength"],
            "source_diversity": diversity,
            "reliability": reliability_score,
            "retrieval_quality": retrieval_quality,
            "cross_source_agreement": agreement,
            "duplicate_penalty": duplicate_penalty,
            "confidence_inputs": confidence_inputs,
            "overall_verification_quality": overall,
        }

    def _risk_analysis(self, claim, processed, sources, quality, mode):
        fake = self.analyze_fake_news(claim) if mode == "news" or processed["intent"] == "news_verification" else {"probability": 0}
        manipulation = self._risk_label(fake.get("probability", 0))
        low_reliability = len([source for source in sources if source.get("reliability", 0) <= 3])
        bias_score = max(0, 100 - quality.get("source_diversity", 0)) + (low_reliability * 10)
        ambiguity_score = 20 if processed.get("complexity") == "Low" else 45 if processed.get("complexity") == "Medium" else 70
        if not any(processed.get("entities", {}).values()):
            ambiguity_score += 15
        return {
            "manipulation_risk": manipulation,
            "bias_risk": self._risk_label(min(100, bias_score)),
            "evidence_coverage": quality.get("coverage_score", 0),
            "claim_ambiguity": self._risk_label(min(100, ambiguity_score)),
        }

    def _confidence(self, scores, evidence_items, quality):
        confidence = self.confidence.calculate_breakdown(**quality.get("confidence_inputs", {
            "evidence_strength": quality.get("evidence_strength_adjusted", quality.get("evidence_strength", 0)),
            "source_credibility": quality.get("reliability", 0),
            "agreement": quality.get("cross_source_agreement", 0),
            "coverage": quality.get("coverage_score", 0),
        }))
        support_items = [item for item in evidence_items if item.get("stance") == "support"]
        contradiction_items = [item for item in evidence_items if item.get("stance") == "contradict"]
        avg_similarity = sum(item.get("combined_similarity", 0) for item in evidence_items) / max(1, len(evidence_items))
        if support_items and not contradiction_items:
            if len(support_items) <= 1:
                confidence["percentage"] = min(confidence["percentage"], max(35, quality.get("coverage_score", 0)))
            if quality.get("coverage_score", 0) < 50:
                confidence["percentage"] = min(confidence["percentage"], 50)
            if avg_similarity < 40:
                confidence["percentage"] = min(confidence["percentage"], 45)
            confidence["level"] = self.confidence.level(confidence["percentage"])
        if evidence_items and confidence["percentage"] == 0:
            confidence["percentage"] = min(44, max(20, quality.get("coverage_score", 0)))
            confidence["level"] = self.confidence.level(confidence["percentage"])
        return confidence

    def _status(self, evidence_items, retrieval):
        if not evidence_items:
            search_reached = self._search_reached(retrieval.get("search_diagnostics", {}))
            if retrieval.get("sources_searched", 0) > 0 or retrieval.get("pages_retrieved", 0) > 0 or search_reached:
                return "UNCERTAIN"
            return "INSUFFICIENT DATA"
        if not any(item.get("stance") in {"support", "contradict"} for item in evidence_items):
            return "UNCERTAIN"
        return "VERIFIED"

    def _determine_verdict(self, scores, evidence_items, confidence=None, quality=None):
        support_score = scores["support"]
        contradiction_score = scores["contradiction"]
        neutral_score = scores["neutral"]
        total = support_score + contradiction_score + neutral_score
        if total <= 0 or not evidence_items:
            return "Insufficient Evidence"
        support_count = len([item for item in evidence_items if item["stance"] == "support"])
        contradiction_count = len([item for item in evidence_items if item["stance"] == "contradict"])
        if support_count == 0 and contradiction_count == 0:
            return "Insufficient Evidence"
        ratio = support_score / max(1.0, support_score + contradiction_score)
        avg_coverage = 0
        avg_similarity = 0
        if evidence_items:
            avg_coverage = sum(item.get("claim_coverage", 0) for item in evidence_items) / len(evidence_items)
            avg_similarity = sum(item.get("combined_similarity", 0) for item in evidence_items) / len(evidence_items)
        confidence_pct = (confidence or {}).get("percentage", 100 if confidence is None else 0)
        high_cred_contradictions = [
            item for item in evidence_items
            if item.get("stance") == "contradict" and item.get("reliability", 0) >= 7 and item.get("evidence_score", 0) >= 45
        ]
        high_cred_support = [
            item for item in evidence_items
            if item.get("stance") == "support" and item.get("reliability", 0) >= 7 and item.get("evidence_score", 0) >= 45
        ]
        if contradiction_count > 0 and support_count == 0 and high_cred_contradictions:
            return "False" if confidence_pct >= 60 else "Mostly False"
        if contradiction_count >= support_count + 2 and high_cred_contradictions:
            return "False" if confidence_pct >= 70 else "Mostly False"
        if contradiction_count == 0 and support_count == 1:
            support_items = [item for item in evidence_items if item.get("stance") == "support"]
            support_coverage = support_items[0].get("claim_coverage", 0) if support_items else 0
            support_similarity = support_items[0].get("combined_similarity", 0) if support_items else 0
            if support_coverage >= 75 and support_similarity >= 55 and confidence_pct >= 55:
                return "Mostly True"
            return "Insufficient Evidence"
        if contradiction_count == 0 and support_count >= 2 and ratio >= 0.82 and avg_coverage >= 70 and confidence_pct >= 85 and (len(high_cred_support) >= 1 or confidence is None):
            return "True"
        if contradiction_count == 0 and support_count >= 2 and ratio >= 0.65 and avg_coverage >= 45 and confidence_pct >= 55:
            return "Mostly True"
        if support_count > contradiction_count and ratio >= 0.85 and avg_coverage >= 55 and confidence_pct >= 60:
            return "Mostly True"
        if support_count >= contradiction_count + 2 and ratio >= 0.72 and avg_coverage >= 45 and confidence_pct >= 55:
            return "Mostly True"
        if support_count > 0 and contradiction_count > 0 and 0.45 <= ratio < 0.62:
            return "Mixed Evidence"
        if contradiction_count > support_count and ratio <= 0.38:
            if contradiction_count >= 2 and ratio <= 0.25:
                return "False"
            return "Mostly False"
        if support_count == 0 and contradiction_count > 0:
            if ratio <= 0.2:
                return "False"
            return "Mostly False"
        return "Mixed Evidence"

    def _verdict_rule_from_verdict(self, verdict):
        mapping = {
            "True": "strong_support",
            "Mostly True": "moderate_support",
            "Mixed Evidence": "balanced_evidence",
            "Mostly False": "moderate_contradiction",
            "False": "strong_contradiction",
            "Unsupported Claim": "search_worked_no_support",
            "Likely Fake Claim": "fabricated_claim_pattern",
            "Fake Claim": "fabricated_claim_pattern",
            "Insufficient Evidence": "insufficient_retrieval_or_readability",
        }
        return mapping.get(verdict, "balanced_evidence")

    def _claim_segments(self, processed):
        """Build claim subject/predicate/object token sets for evidence filtering."""
        claim_structure = processed.get("nlp_pipeline", {}).get("claim_structure", {})
        return {
            "subject": self._expanded_terms(claim_structure.get("subject", "")),
            "predicate": self._expanded_terms(claim_structure.get("predicate", "")),
            "object": self._expanded_terms(claim_structure.get("object", "")),
        }

    def _expanded_terms(self, text):
        terms = set(tokenize_words(text))
        expanded = set(terms)
        irregular = {
            "has": {"have", "had"},
            "have": {"has", "had"},
            "is": {"are", "be", "was"},
            "are": {"is", "be", "were"},
            "was": {"is", "be", "were"},
            "were": {"are", "be", "was"},
        }
        for term in terms:
            expanded.update(irregular.get(term, set()))
            if term.endswith("s") and len(term) > 3:
                expanded.add(term[:-1])
            else:
                expanded.add(term + "s")
        return expanded

    def _claim_entity_terms(self, processed):
        """Collect entity tokens for stricter evidence matching."""
        entity_terms = set()
        entities = processed.get("entities", {})
        for group in entities.values():
            for item in group:
                entity_terms.update(tokenize_words(item))
        return {token for token in entity_terms if token}

    def _claim_rare_terms(self, processed):
        """Collect distinctive non-entity claim tokens."""
        claim_tokens = set(processed.get("tokens", []))
        claim_segments = self._claim_segments(processed)
        common_terms = set().union(*claim_segments.values()) if claim_segments else set()
        common_terms.update(self._claim_entity_terms(processed))
        rare_terms = {token for token in claim_tokens if token not in common_terms}
        return {token for token in rare_terms if token}

    def _source_record(self, url, title, credibility, stance, evidence, page):
        return {
            "url": url,
            "title": clean_text(title),
            "domain": credibility.get("domain") or domain_from_url(url),
            "reliability": credibility.get("reliability", 0),
            "credibility_score": credibility.get("normalized", 0),
            "credibility_tier": credibility.get("tier", ""),
            "reliability_reason": credibility.get("reason", ""),
            "stance": stance,
            "evidence": evidence,
            "status_code": page.get("status_code"),
            "text_length": page.get("text_length", 0),
            "extraction_quality": page.get("extraction_quality", 0),
            "retrieved": page.get("retrieved", False),
            "accepted": page.get("accepted", False),
            "rejection_reason": page.get("rejection_reason", ""),
        }

    def _citations(self, sources):
        citations = []
        for index, source in enumerate(sources, start=1):
            if not source.get("url"):
                continue
            citations.append(
                {
                    "number": index,
                    "title": source.get("title", "") or source.get("domain", ""),
                    "url": source.get("url", ""),
                    "domain": source.get("domain", ""),
                }
            )
        return citations

    def _explain(self, verdict, confidence, scores, evidence_items, quality, reliability=None, source_clusters=None):
        supporting = len([item for item in evidence_items if item["stance"] == "support"])
        contradicting = len([item for item in evidence_items if item["stance"] == "contradict"])
        neutral = len([item for item in evidence_items if item["stance"] == "neutral"])
        agreement = int(round((max(supporting, contradicting) / max(1, supporting + contradicting + neutral)) * 100))
        top_support = self._top_evidence_summary(evidence_items, "support")
        top_contradiction = self._top_evidence_summary(evidence_items, "contradict")
        reliability = reliability or {}
        source_clusters = source_clusters or []
        breakdown = confidence.get("breakdown", {})
        reasons = [
            "The claim was compared against extracted evidence from {0} independent source cluster(s).".format(len(source_clusters)),
            "{0} evidence item(s) support the claim, {1} contradict it, and {2} are neutral.".format(supporting, contradicting, neutral),
            "Source agreement is {0}% and average independent source credibility is {1}/100.".format(
                agreement,
                reliability.get("independent_weighted_reliability_score", quality.get("reliability", 0)),
            ),
            "Confidence formula: 30% evidence strength + 25% source credibility + 25% agreement + 20% claim coverage.",
            "Confidence inputs: evidence strength {0}, credibility {1}, agreement {2}, coverage {3}.".format(
                breakdown.get("evidence_strength", quality.get("evidence_strength_adjusted", 0)),
                breakdown.get("source_credibility", quality.get("reliability", 0)),
                breakdown.get("cross_source_agreement", quality.get("cross_source_agreement", 0)),
                breakdown.get("claim_coverage", quality.get("coverage_score", 0)),
            ),
        ]
        if top_support:
            reasons.append("Strongest supporting evidence: " + top_support)
        if top_contradiction:
            reasons.append("Strongest contradicting evidence: " + top_contradiction)
        if scores["support"] > scores["contradiction"]:
            reasons.append("Most highly-ranked weighted evidence supports the claim.")
        elif scores["contradiction"] > scores["support"]:
            reasons.append("The strongest weighted evidence contradicts the claim.")
        else:
            reasons.append("The evidence is balanced or weak, so the verdict remains cautious.")
        if quality.get("duplicate_penalty", 0):
            reasons.append("Confidence was reduced because duplicate or non-independent source patterns were detected.")
        return "\n".join(
            [
                "VERDICT: " + verdict,
                "Confidence: {0}% ({1})".format(confidence["percentage"], confidence["level"]),
                "Reasoning Engine:",
            ]
            + reasons
        )

    def _top_evidence_summary(self, evidence_items, stance):
        candidates = [item for item in evidence_items if item.get("stance") == stance]
        if not candidates:
            return ""
        best = max(candidates, key=lambda item: (item.get("weighted_score", 0), item.get("reliability", 0)))
        sentence = clean_text(best.get("sentence", ""))
        if len(sentence) > 220:
            sentence = sentence[:217] + "..."
        return "{0} ({1}, reliability {2}/10)".format(sentence, best.get("domain", "unknown"), best.get("reliability", 0))

    def _insufficient_data_explanation(self, retrieval):
        diagnostics = retrieval.get("search_diagnostics", {})
        causes = [
            "Internet unavailable" if diagnostics.get("status_code") is None else "",
            "Search endpoint blocked or returned no parseable results" if retrieval.get("sources_searched", 0) == 0 else "",
            "Search results did not contain enough directly relevant evidence" if retrieval.get("sources_searched", 0) > 0 else "",
            "Query may be too recent, fictional, local, or not covered by reliable indexed sources",
            "Source pages may be unavailable, blocked, too short, or unrelated to the exact claim",
        ]
        lines = [
            "STATUS: INSUFFICIENT DATA",
            "Verdict: Insufficient Evidence",
            "Reason: The claim could not be verified because reliable evidence could not be retrieved.",
            "Search URL: " + str(diagnostics.get("search_url", "")),
            "Search status code: " + str(diagnostics.get("status_code")),
            "Search error: " + (diagnostics.get("error") or "None"),
            "Sources searched: {0}".format(retrieval.get("sources_searched", 0)),
            "Pages retrieved: {0}".format(retrieval.get("pages_retrieved", 0)),
            "Pages rejected: {0}".format(retrieval.get("pages_rejected", 0)),
            "Extraction success rate: {0}%".format(retrieval.get("extraction_success_rate", 0)),
            "Possible Causes:",
        ]
        lines.extend("- " + cause for cause in causes if cause)
        lines.append("Recommended Action: Try again later, rephrase the claim, or verify against known official sources.")
        return "\n".join(lines)

    def _unsupported_claim_explanation(self, retrieval):
        diagnostics = retrieval.get("search_diagnostics", {})
        lines = [
            "VERDICT: Unsupported Claim",
            "Confidence: 0% (Very Low)",
            "Reason: Live search completed, but no credible source supported the claim.",
            "Search URL: " + str(diagnostics.get("search_url", "")),
            "Search status code: " + str(diagnostics.get("status_code")),
            "Sources searched: {0}".format(retrieval.get("sources_searched", 0)),
            "Pages retrieved: {0}".format(retrieval.get("pages_retrieved", 0)),
            "Pages accepted for NLP analysis: {0}".format(retrieval.get("sources_successfully_retrieved", 0)),
            "Extraction success rate: {0}%".format(retrieval.get("extraction_success_rate", 0)),
            "Reasoning Engine:",
            "- Search completed but did not produce a credible supporting source.",
            "- The retrieved material did not establish the claim as true.",
            "- The safest evidence-based conclusion is Unsupported Claim.",
            "Recommended Action: Verify against an official source or a higher-credibility reference.",
        ]
        attempts = diagnostics.get("fallback_attempts", [])
        if attempts:
            lines.append("Fallback Search Attempts:")
            for attempt in attempts:
                lines.append(
                    "- {0} | status={1} | results={2}".format(
                        attempt.get("query", ""),
                        attempt.get("status_code"),
                        attempt.get("results_returned", 0),
                    )
                )
        return "\n".join(lines)

    def _verification_diagnostics(self, claim, processed, search_bundle, retrieval, reliability, quality, evidence_items, verdict_rule):
        diagnostics = search_bundle.get("diagnostics", {})
        fallback_attempts = diagnostics.get("fallback_attempts", [])
        page_diagnostics = retrieval.get("page_diagnostics", [])
        rejection_reasons = []
        for page in page_diagnostics:
            reason = page.get("rejection_reason", "")
            if reason:
                rejection_reasons.append(reason)
        return {
            "original_claim": claim,
            "normalized_claim": processed.get("normalized_text", claim),
            "generated_queries": processed.get("query_variants", []),
            "fallback_queries_used": fallback_attempts,
            "search_result_count": retrieval.get("sources_searched", 0),
            "retrieved_page_count": retrieval.get("pages_retrieved", 0),
            "readable_page_count": retrieval.get("sources_successfully_retrieved", 0),
            "rejected_page_count": retrieval.get("pages_rejected", 0),
            "rejection_reasons": self._unique(rejection_reasons + [diagnostics.get("error", "")]),
            "extracted_evidence_count": len(evidence_items),
            "support_count": len([item for item in evidence_items if item.get("stance") == "support"]),
            "contradiction_count": len([item for item in evidence_items if item.get("stance") == "contradict"]),
            "neutral_count": len([item for item in evidence_items if item.get("stance") == "neutral"]),
            "credibility_distribution": reliability.get("tier_distribution", {}),
            "coverage_score": quality.get("coverage_score", 0),
            "agreement_score": quality.get("cross_source_agreement", 0),
            "final_verdict_rule": verdict_rule,
        }

    def _search_reached(self, diagnostics):
        return any(
            diagnostics.get(key) == 200
            for key in ("status_code", "fallback_status_code", "secondary_status_code")
        )

    def _likely_fake_claim_explanation(self, retrieval, assessment):
        diagnostics = retrieval.get("search_diagnostics", {})
        label = assessment.get("label") or "Likely Fake Claim"
        confidence = assessment.get("confidence", {"percentage": 0, "level": "Very Low"})
        lines = [
            "VERDICT: " + label,
            "Confidence: {0}% ({1})".format(confidence["percentage"], confidence["level"]),
            "Reason: Live web search was attempted, but no credible source was found to confirm the claim and the wording matches unsupported fabricated-claim patterns.",
            "Search URL: " + str(diagnostics.get("search_url", "")),
            "Search status code: " + str(diagnostics.get("status_code")),
            "Sources searched: {0}".format(retrieval.get("sources_searched", 0)),
            "Pages retrieved: {0}".format(retrieval.get("pages_retrieved", 0)),
            "Pages accepted for NLP analysis: {0}".format(retrieval.get("sources_successfully_retrieved", 0)),
            "Extraction success rate: {0}%".format(retrieval.get("extraction_success_rate", 0)),
            "Reasoning Engine:",
        ]
        for reason in assessment.get("reasons", [])[:6]:
            lines.append("- " + reason)
        lines.append("- This verdict is based on failed corroboration plus suspicious claim patterns, not on a direct source explicitly denying the claim.")
        lines.append("Recommended Action: Treat the claim as likely fake unless a credible source, official statement, or verified report is found.")
        attempts = diagnostics.get("fallback_attempts", [])
        if attempts:
            lines.append("Fallback Search Attempts:")
            for attempt in attempts:
                lines.append(
                    "- {0} | status={1} | results={2}".format(
                        attempt.get("query", ""),
                        attempt.get("status_code"),
                        attempt.get("results_returned", 0),
                    )
                )
        return "\n".join(lines)

    def _build_answer(self, status, verdict, confidence, explanation, evidence_items):
        if status == "INSUFFICIENT DATA":
            return explanation
        best = evidence_items[:3]
        evidence_text = " ".join(item["sentence"] for item in best)
        if evidence_text:
            return explanation + "\nKey evidence: " + evidence_text
        return explanation

    def _risk_label(self, score):
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    def _unique(self, values):
        seen = set()
        unique_values = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values
