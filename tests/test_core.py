"""Offline tests for the standard-library-only fact-checking modules."""

import unittest

from confidence import ConfidenceEngine
from contradiction_detector import ContradictionDetector
from credibility import CredibilityAnalyzer
from database import DatabaseManager
from evidence_extractor import EvidenceExtractor
from nlp_evaluation import NLPEvaluator
from query_processor import QueryProcessor
from search_engine import SearchEngine
from verifier import FactCheckingEngine
from web_scraper import WebScraper


class CoreModuleTests(unittest.TestCase):
    def test_query_processing_extracts_keywords(self):
        processor = QueryProcessor()
        result = processor.process("Did the WHO declare COVID-19 a pandemic in 2020?")
        self.assertIn("covid", result["tokens"])
        self.assertTrue(result["keywords"])
        self.assertEqual(result["question_type"], "temporal_question")
        self.assertIn("organizations", result["entities"])
        self.assertNotIn("Did", result["entities"]["people"])
        self.assertIn("declare", result["nlp_pipeline"]["lemmas"])
        self.assertIn("category", result)

    def test_html_text_extraction_removes_script(self):
        scraper = WebScraper()
        page = scraper.extract_text("<html><title>T</title><script>bad()</script><p>Visible text here.</p></html>")
        self.assertEqual(page["title"], "T")
        self.assertIn("Visible text here", page["text"])
        self.assertNotIn("bad", page["text"])

    def test_credibility_scoring(self):
        analyzer = CredibilityAnalyzer()
        self.assertEqual(analyzer.score("https://example.gov/report")["reliability"], 10)
        self.assertEqual(analyzer.score("https://example.edu/research")["reliability"], 9)
        self.assertEqual(analyzer.score("https://example.edu/research")["tier"], "Tier 1")

    def test_evidence_extraction(self):
        extractor = EvidenceExtractor()
        evidence = extractor.extract(
            "The World Health Organization declared COVID-19 a pandemic in 2020. Another sentence.",
            ["world", "health", "organization", "covid", "pandemic"],
            "WHO declared COVID-19 a pandemic in 2020",
        )
        self.assertTrue(evidence)
        self.assertIn("pandemic", evidence[0]["sentence"].lower())
        self.assertIn("jaccard_similarity", evidence[0])
        self.assertIn("cosine_similarity", evidence[0])
        self.assertGreater(evidence[0]["combined_similarity"], 0)

    def test_contradiction_detection(self):
        detector = ContradictionDetector()
        stance = detector.classify("The vaccine is effective", "The report says the vaccine is not effective.")
        self.assertEqual(stance, "contradict")

    def test_numeric_contradiction_detection(self):
        detector = ContradictionDetector()
        stance = detector.classify("The mission launched in 2020", "The mission launched in 2021.")
        self.assertEqual(stance, "contradict")

    def test_negation_emphasis_is_not_contradiction(self):
        detector = ContradictionDetector()
        stance = detector.classify("Karachi is Pakistan's biggest city", "Karachi isn't just Pakistan's biggest city; it is a major port city.")
        self.assertEqual(stance, "support")

    def test_myth_framing_contradicts_claim(self):
        detector = ContradictionDetector()
        self.assertEqual(
            detector.classify("Lightning never strikes the same place twice", "Myth: Lightning never strikes twice in the same place."),
            "contradict",
        )
        self.assertEqual(
            detector.classify("Lightning never strikes the same place twice", "Fact: Lightning can strike the same place multiple times."),
            "contradict",
        )

    def test_confidence_engine(self):
        confidence = ConfidenceEngine().calculate(80, 20)
        self.assertEqual(confidence["percentage"], 80)
        self.assertEqual(confidence["level"], "High")

    def test_confidence_breakdown_formula(self):
        confidence = ConfidenceEngine().calculate_breakdown(80, 90, 75, 70)
        self.assertEqual(confidence["percentage"], 79)
        self.assertEqual(confidence["breakdown"]["source_credibility"], 90)

    def test_search_failure_returns_no_local_fallback(self):
        engine = SearchEngine(max_results=5)
        engine._download = lambda url: (_ for _ in ()).throw(OSError("blocked"))
        bundle = engine.search_with_diagnostics("Karachi is the largest city in Pakistan")
        self.assertEqual(bundle["results"], [])
        self.assertNotIn("offline_fallback_used", bundle["diagnostics"])
        self.assertIn("blocked", bundle["diagnostics"]["error"])

    def test_verdict_taxonomy(self):
        engine = FactCheckingEngine(database_path=":memory:")
        self.assertEqual(
            engine._determine_verdict(
                {"support": 100, "contradiction": 0, "neutral": 0},
                [
                    {"stance": "support", "claim_coverage": 70, "combined_similarity": 68},
                    {"stance": "support", "claim_coverage": 72, "combined_similarity": 70},
                ],
            ),
            "True",
        )
        self.assertEqual(
            engine._determine_verdict({"support": 70, "contradiction": 20, "neutral": 10}, [{"stance": "support"}, {"stance": "contradict"}]),
            "Mixed Evidence",
        )

    def test_neutral_only_evidence_does_not_count_as_verified(self):
        engine = FactCheckingEngine(database_path=":memory:")
        status = engine._status(
            [{"stance": "neutral"}],
            {
                "search_diagnostics": {"status_code": 200},
                "sources_searched": 1,
                "pages_retrieved": 1,
            },
        )
        self.assertEqual(status, "UNCERTAIN")

    def test_fake_news_rules(self):
        engine = FactCheckingEngine(database_path=":memory:")
        report = engine.analyze_fake_news("BREAKING!!! SHOCKING secret cure exposed!!!")
        self.assertGreaterEqual(report["probability"], 40)

    def test_claim_representation_and_query_variants(self):
        processor = QueryProcessor()
        result = processor.process("Earth has two moons")
        representation = result["claim_representation"]
        self.assertIn("scientific claim", representation["claim_types"])
        self.assertIn("numerical claim", representation["claim_types"])
        self.assertTrue(any(item["type"] == "contradiction" for item in result["query_variants"]))

    def test_news_query_generation_is_generic(self):
        processor = QueryProcessor()
        result = processor.process("The central bank reduced its growth forecast to 2.5 percent, according to its latest report.")
        query_types = {item["type"] for item in result["query_variants"]}
        joined_queries = " ".join(item["query"].lower() for item in result["query_variants"])
        self.assertIn("event_news", query_types)
        self.assertIn("major_news", query_types)
        self.assertIn("2.5", joined_queries)
        self.assertIn("forecast", joined_queries)

    def test_open_domain_query_generation_includes_dynamic_variants(self):
        processor = QueryProcessor()
        result = processor.process("The chemical symbol for gold is Au.")
        query_types = {item["type"] for item in result["query_variants"]}
        joined_queries = " ".join(item["query"].lower() for item in result["query_variants"])

        self.assertIn("exact_claim", query_types)
        self.assertIn("simplified_claim", query_types)
        self.assertIn("entity_relation", query_types)
        self.assertIn("verification_true", query_types)
        self.assertIn("verification_false", query_types)
        self.assertIn("fact_check", query_types)
        self.assertIn("official_source", query_types)
        self.assertIn("credible_source", query_types)
        self.assertIn("contradiction", query_types)
        self.assertIn("evidence", query_types)
        self.assertIn("gold", joined_queries)
        self.assertGreaterEqual(len(result["query_variants"]), 8)

    def test_search_worked_but_no_support_becomes_unsupported_claim(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": 200,
                "response_bytes": 2048,
                "raw_results_parsed": 0,
                "results_returned": 0,
                "error": "",
                "fallback_attempts": [{"query": query, "status_code": 200, "results_returned": 0}],
            },
        }

        result = engine.verify_claim("The chemical symbol for gold is Au.")
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["verdict"], "Unsupported Claim")

    def test_verification_diagnostics_expose_pipeline_summary(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": 200,
                "response_bytes": 2048,
                "raw_results_parsed": 0,
                "results_returned": 0,
                "error": "",
                "fallback_attempts": [{"query": query, "status_code": 200, "results_returned": 0}],
            },
        }

        result = engine.verify_claim("Lightning never strikes the same place twice.")
        diagnostics = result.get("verification_diagnostics", {})
        self.assertEqual(diagnostics.get("original_claim"), "Lightning never strikes the same place twice.")
        self.assertIn("normalized_claim", diagnostics)
        self.assertIn("generated_queries", diagnostics)
        self.assertIn("fallback_queries_used", diagnostics)
        self.assertIn("final_verdict_rule", diagnostics)

    def test_semantic_numeric_contradiction(self):
        detector = ContradictionDetector()
        self.assertEqual(detector.classify("Earth has two moons", "Earth has one natural satellite."), "contradict")

    def test_database_save_and_history(self):
        database = DatabaseManager(":memory:")
        result = {
            "query": "Sample claim",
            "status": "INSUFFICIENT DATA",
            "verdict": "UNCERTAIN",
            "confidence": {"percentage": None, "level": "Not Available"},
            "processed_query": {"category": "General", "keywords": ["sample"]},
            "quality_metrics": {"overall_verification_quality": 0},
            "explanation": "No evidence.",
            "sources": [],
            "citations": [],
        }
        fact_check_id = database.save_fact_check(result)
        self.assertEqual(fact_check_id, 1)
        self.assertEqual(len(database.list_history()), 1)
        self.assertEqual(database.analytics_summary()["total_fact_checks"], 1)

    def test_insufficient_data_does_not_report_zero_confidence(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": None,
                "response_bytes": 0,
                "raw_results_parsed": 0,
                "results_returned": 0,
                "error": "offline",
            },
        }
        result = engine.verify_claim("Trump says Beirut attack by Israel today should not have happened.")
        self.assertEqual(result["status"], "INSUFFICIENT DATA")
        self.assertIsNone(result["confidence"]["percentage"])
        self.assertIn("Search URL", result["explanation"])

    def test_weak_location_match_is_not_mostly_true(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [
                {
                    "url": "https://zmescience.com/story/cats-karachi",
                    "title": "Cats in Karachi",
                }
            ],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": 200,
                "response_bytes": 1024,
                "raw_results_parsed": 1,
                "results_returned": 1,
                "error": "",
            },
        }
        engine.scraper.fetch = lambda url: {
            "title": "Cats in Karachi",
            "text": (
                "Two wild cats thought to be disappearing in Pakistan just reappeared on camera in Karachi. "
                "The park is threatened by sand and gravel mining driven by development in the nearby megacity of Karachi."
            ),
            "accepted": True,
            "retrieved": True,
            "status_code": 200,
            "text_length": 180,
            "extraction_quality": 55,
            "response_bytes": 1800,
        }

        result = engine.verify_claim("scientists discovered flying cats in karachi")
        self.assertNotEqual(result["verdict"], "Mostly True")
        self.assertIn(result["verdict"], {"Mostly False", "False", "Mixed Evidence", "Insufficient Evidence", "Likely Fake Claim", "Fake Claim"})

    def test_unsupported_extraordinary_claim_is_marked_likely_fake(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": 200,
                "response_bytes": 2048,
                "raw_results_parsed": 0,
                "results_returned": 0,
                "error": "",
                "fallback_attempts": [
                    {"query": query, "status_code": 200, "results_returned": 0},
                    {"query": query + " reliable source", "status_code": 200, "results_returned": 0},
                ],
            },
        }
        result = engine.verify_claim("scientists discovered flying cats in karachi")
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertIn(result["verdict"], {"Likely Fake Claim", "Fake Claim"})
        self.assertIn("unsupported fabricated-claim patterns", result["explanation"])

    def test_plain_unverified_claim_remains_insufficient_evidence(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": 200,
                "response_bytes": 2048,
                "raw_results_parsed": 0,
                "results_returned": 0,
                "error": "",
                "fallback_attempts": [
                    {"query": query, "status_code": 200, "results_returned": 0},
                ],
            },
        }
        result = engine.verify_claim("the city museum opened in 1967")
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["verdict"], "Unsupported Claim")

    def test_search_title_fallback_can_supply_contradiction(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [
                {
                    "url": "https://example.edu/earth-moons",
                    "title": "Does Earth have two moons? What is a quasi-moon?",
                    "domain": "example.edu",
                }
            ],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": 200,
                "response_bytes": 2048,
                "raw_results_parsed": 1,
                "results_returned": 1,
                "error": "",
            },
        }
        engine.scraper.fetch = lambda url: {
            "title": "",
            "text": "",
            "accepted": False,
            "retrieved": False,
            "status_code": 403,
            "text_length": 0,
            "extraction_quality": 0,
            "response_bytes": 0,
            "rejection_reason": "HTTP Error 403: Forbidden",
        }
        result = engine.verify_claim("Earth has two moons")
        self.assertGreaterEqual(result["contradicting_sources"], 1)
        self.assertIn(result["verdict"], {"Mostly False", "False", "Mixed Evidence"})

    def test_search_result_ranking_penalizes_dictionary_pages_for_news(self):
        engine = FactCheckingEngine(database_path=":memory:")
        processed = engine.query_processor.process("A company announced a new product launch according to Reuters.")
        results = [
            {
                "title": "ANNOUNCED definition and meaning",
                "url": "https://dictionary.example/announced",
                "domain": "dictionary.example",
                "snippet": "Definition and pronunciation of announced.",
                "query_type": "exact_claim",
            },
            {
                "title": "Company announced new product launch",
                "url": "https://www.reuters.com/business/company-product-launch",
                "domain": "reuters.com",
                "snippet": "The company announced a new product launch, according to Reuters.",
                "query_type": "major_news",
            },
        ]
        ranked = engine._rank_search_results(results, processed)
        self.assertEqual(ranked[0]["domain"], "reuters.com")

    def test_search_metadata_can_support_blocked_news_page(self):
        engine = FactCheckingEngine(database_path=":memory:")
        engine.search_engine.search_with_diagnostics = lambda query, max_results=None: {
            "results": [
                {
                    "url": "https://news.example/world/event",
                    "title": "Central bank reduces growth forecast to 2.5 percent",
                    "domain": "news.example",
                    "snippet": "The central bank reduced its economic growth forecast to 2.5 percent, according to its latest report.",
                }
            ],
            "diagnostics": {
                "query": query,
                "search_url": "https://example.com/search?q=test",
                "status_code": 200,
                "response_bytes": 2048,
                "raw_results_parsed": 1,
                "results_returned": 1,
                "error": "",
            },
        }
        engine.scraper.fetch = lambda url: {
            "title": "",
            "text": "",
            "accepted": False,
            "retrieved": False,
            "status_code": 403,
            "text_length": 0,
            "extraction_quality": 0,
            "response_bytes": 0,
            "rejection_reason": "HTTP Error 403: Forbidden",
        }
        result = engine.verify_claim("The central bank reduced its economic growth forecast to 2.5 percent, according to its latest report.", mode="news")
        self.assertGreaterEqual(result["supporting_sources"], 1)
        self.assertIn(result["status"], {"VERIFIED", "UNCERTAIN"})

    def test_weak_location_match_is_rejected_as_relevant_evidence(self):
        engine = FactCheckingEngine(database_path=":memory:")
        processed = engine.query_processor.process("scientists discovered flying cats in karachi")
        candidate = {
            "sentence": "Two wild cats thought to be disappearing in Pakistan just reappeared on camera in Karachi.",
            "context": "Two wild cats thought to be disappearing in Pakistan just reappeared on camera in Karachi. The park is threatened by sand and gravel mining driven by development in the nearby megacity of Karachi.",
            "claim_coverage": 50,
        }

        self.assertFalse(engine._candidate_relevant(candidate, processed))

    def test_generic_science_match_without_rare_anchor_is_rejected(self):
        engine = FactCheckingEngine(database_path=":memory:")
        processed = engine.query_processor.process("scientists discovered flying cats in karachi")
        candidate = {
            "sentence": "French scientists wanted to use larger mammals and chose cats since they already had significant neurological data.",
            "context": "French scientists wanted to use larger mammals and chose cats since they already had significant neurological data. Mission selection continued later.",
            "claim_coverage": 40,
        }

        self.assertFalse(engine._candidate_relevant(candidate, processed))

    def test_scientific_claim_rejects_sports_metaphor(self):
        engine = FactCheckingEngine(database_path=":memory:")
        processed = engine.query_processor.process("Lightning never strikes the same place twice.")
        candidate = {
            "sentence": '"Lightning struck twice in the same place" as the Warriors built another NBA dynasty.',
            "context": '"Lightning struck twice in the same place" as the Warriors built another NBA dynasty.',
            "claim_coverage": 80,
        }

        self.assertFalse(engine._candidate_relevant(candidate, processed))

    def test_offline_nlp_evaluation(self):
        evaluator = NLPEvaluator()
        metrics = evaluator.evaluate(
            [
                {
                    "text": "Did the WHO declare COVID-19 a pandemic in 2020?",
                    "expected_question_type": "temporal_question",
                    "expected_category": "Health",
                    "expected_keywords": "declare|covid|pandemic|2020",
                    "expected_entities": "WHO|COVID-19|2020",
                }
            ]
        )
        self.assertEqual(metrics["examples"], 1)
        self.assertEqual(metrics["question_type_accuracy"], 100)
        self.assertGreaterEqual(metrics["keyword_recall"], 75)


if __name__ == "__main__":
    unittest.main()
