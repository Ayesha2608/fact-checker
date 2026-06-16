"""Professional reporting helpers for console, text, CSV, and analytics output."""

import csv
from pathlib import Path


class ReportGenerator:
    """Generate platform-grade verification and analytics reports."""

    def text_report(self, result):
        """Build a complete verification dashboard report."""
        processed = result.get("processed_query", {})
        retrieval = result.get("retrieval_analysis", {})
        reliability = result.get("source_reliability_analysis", {})
        quality = result.get("quality_metrics", {})
        risk = result.get("risk_analysis", {})
        confidence = result.get("confidence", {})
        lines = [
            "=" * 78,
            "AI-POWERED EVIDENCE ANALYSIS AND FACT VERIFICATION PLATFORM",
            "=" * 78,
            "Claim ID: " + result.get("claim_id", "Pending"),
            "Timestamp: " + result.get("timestamp", ""),
            "Status: " + result.get("status", ""),
            "Claim: " + result.get("query", ""),
            "",
            "QUERY ANALYSIS",
            "Detected Keywords: " + self._join(processed.get("keywords", [])),
            "Keyword Importance Ranking: " + self._keyword_scores(processed.get("keyword_scores", [])),
            "Claim Category: " + processed.get("category", "General"),
            "Query Complexity: " + processed.get("complexity", "Unknown"),
            "Sentiment: " + processed.get("sentiment", {}).get("label", "Neutral"),
            "Named Entities Detected:",
        ]
        lines.extend(self._entity_lines(processed.get("entities", {})))
        lines.extend(self._nlp_pipeline_block(processed.get("pipeline_summary", {})))
        lines.extend(
            [
                "",
                "RETRIEVAL ANALYSIS",
                "Exact Search URL: " + str(retrieval.get("search_diagnostics", {}).get("search_url", "")),
                "Search Status Code: " + str(retrieval.get("search_diagnostics", {}).get("status_code")),
                "Sources Searched: " + str(retrieval.get("sources_searched", 0)),
                "Sources Successfully Retrieved: " + str(retrieval.get("sources_successfully_retrieved", 0)),
                "Failed Sources: " + str(retrieval.get("failed_sources", 0)),
                "Domains Contacted: " + self._join(retrieval.get("domains_contacted", [])),
                "Domains Retrieved: " + self._join(retrieval.get("domains_retrieved", [])),
                "Average Retrieval Quality: {0}%".format(retrieval.get("average_retrieval_quality", 0)),
                "Extraction Success Rate: {0}%".format(retrieval.get("extraction_success_rate", 0)),
                "",
                "SOURCE RELIABILITY ANALYSIS",
                "Government Sources: " + str(reliability.get("government_sources", 0)),
                "Educational Sources: " + str(reliability.get("educational_sources", 0)),
                "Research Sources: " + str(reliability.get("research_sources", 0)),
                "News Sources: " + str(reliability.get("news_sources", 0)),
                "Other Sources: " + str(reliability.get("other_sources", 0)),
                "Weighted Reliability Score: {0}/100".format(reliability.get("weighted_reliability_score", 0)),
                "",
                "EVIDENCE ANALYSIS",
                "Supporting Evidence: " + str(result.get("supporting_sources", 0)),
                "Contradicting Evidence: " + str(result.get("contradicting_sources", 0)),
                "Neutral Evidence: " + str(result.get("neutral_sources", 0)),
            ]
        )
        lines.extend(self._evidence_block("Top Supporting Statements", result.get("evidence", []), "support"))
        lines.extend(self._evidence_block("Top Contradicting Statements", result.get("evidence", []), "contradict"))
        lines.extend(
            [
                "",
                "VERDICT",
                result.get("verdict", "Insufficient Evidence"),
                "Confidence: " + self._confidence_text(confidence),
                "Confidence Level: " + confidence.get("level", "Not Available"),
                "",
                "REASONING ENGINE",
                result.get("explanation", ""),
                "",
                "RISK ANALYSIS",
                "Manipulation Risk: " + risk.get("manipulation_risk", "Low"),
                "Bias Risk: " + risk.get("bias_risk", "Low"),
                "Evidence Coverage: {0}%".format(risk.get("evidence_coverage", 0)),
                "Claim Ambiguity: " + risk.get("claim_ambiguity", "Low"),
                "",
                "FACT-CHECK QUALITY SCORE",
                "Coverage Score: {0}%".format(quality.get("coverage_score", 0)),
                "Evidence Strength: {0}%".format(quality.get("evidence_strength", 0)),
                "Source Diversity: {0}%".format(quality.get("source_diversity", 0)),
                "Reliability: {0}%".format(quality.get("reliability", 0)),
                "Overall Verification Quality: {0}%".format(quality.get("overall_verification_quality", 0)),
            ]
        )
        if result.get("status") == "INSUFFICIENT DATA":
            lines.extend(self._diagnostics_block(retrieval))
        lines.extend(["", "CITATIONS"])
        for citation in result.get("citations", []):
            lines.append("[{0}] {1}".format(citation["number"], citation["title"]))
            lines.append(citation["url"])
        fake_report = result.get("fake_news_report") or {}
        if fake_report:
            lines.extend(["", "FAKE NEWS HEURISTIC ANALYSIS"])
            lines.append("Fake-news probability: {0}% ({1})".format(fake_report["probability"], fake_report["level"]))
            for reason in fake_report.get("reasons", []):
                lines.append("- " + reason)
        return "\n".join(lines)

    def analytics_report(self, summary):
        """Build an admin analytics dashboard."""
        lines = [
            "=" * 78,
            "ANALYTICS DASHBOARD",
            "=" * 78,
            "Total Fact Checks: " + str(summary.get("total_fact_checks", 0)),
            "Average Confidence: {0}%".format(summary.get("average_confidence", 0)),
            "Average Quality Score: {0}%".format(summary.get("average_quality", 0)),
            "",
            "Most Common Categories:",
        ]
        lines.extend(self._tuple_lines(summary.get("most_common_categories", [])))
        lines.append("")
        lines.append("Top Sources:")
        lines.extend(self._tuple_lines(summary.get("top_sources", [])))
        lines.append("")
        lines.append("Most Frequent Keywords:")
        lines.extend(self._tuple_lines(summary.get("most_frequent_keywords", [])))
        lines.append("")
        lines.append("Verification Statistics:")
        lines.extend(self._tuple_lines(summary.get("verification_statistics", [])))
        lines.append("")
        lines.append("Daily Usage:")
        lines.extend(self._tuple_lines(summary.get("daily_usage", [])))
        return "\n".join(lines)

    def nlp_evaluation_report(self, metrics):
        """Build an offline NLP evaluation report for viva/demo use."""
        lines = [
            "=" * 78,
            "OFFLINE NLP PIPELINE EVALUATION",
            "=" * 78,
            "Examples: " + str(metrics.get("examples", 0)),
            "Question Type Accuracy: {0}%".format(metrics.get("question_type_accuracy", 0)),
            "Category Accuracy: {0}%".format(metrics.get("category_accuracy", 0)),
            "Keyword Recall: {0}%".format(metrics.get("keyword_recall", 0)),
            "Entity Recall: {0}%".format(metrics.get("entity_recall", 0)),
            "",
            "Per-Example Diagnostics:",
        ]
        for item in metrics.get("details", []):
            lines.append("- " + item.get("text", ""))
            lines.append(
                "  question: expected={0}, predicted={1}; category: expected={2}, predicted={3}; keyword_recall={4}%; entity_recall={5}%".format(
                    item.get("expected_question_type", ""),
                    item.get("predicted_question_type", ""),
                    item.get("expected_category", ""),
                    item.get("predicted_category", ""),
                    item.get("keyword_recall", 0),
                    item.get("entity_recall", 0),
                )
            )
        return "\n".join(lines)

    def export_history_csv(self, rows, path):
        """Export history rows returned by DatabaseManager to CSV."""
        target = Path(path)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "query", "date", "category", "status", "verdict", "confidence", "quality_score"])
            for row in rows:
                writer.writerow(row)
        return str(target)

    def export_full_report(self, result, path):
        """Write a full text verification report."""
        target = Path(path)
        target.write_text(self.text_report(result), encoding="utf-8")
        return str(target)

    def _entity_lines(self, entities):
        lines = []
        for label in ("people", "organizations", "countries", "cities", "dates", "numbers", "medical_terms"):
            lines.append("{0}: {1}".format(label.title(), self._join(entities.get(label, []))))
        return lines

    def _nlp_pipeline_block(self, summary):
        if not summary:
            return []
        stages = summary.get("stages", [])
        return [
            "",
            "CORE NLP PIPELINE ARTIFACTS",
            "Stages: " + self._join(stages),
            "Lemmas: " + self._join(summary.get("lemmas", [])),
            "Bigrams: " + self._join(summary.get("bigrams", [])),
            "Noun Phrases: " + self._join(summary.get("noun_phrases", [])),
            "Verb Phrases: " + self._join(summary.get("verb_phrases", [])),
            "Claim Structure: " + str(summary.get("claim_structure", {})),
            "Lexical Diversity: {0}%".format(summary.get("lexical_diversity", 0)),
            "Lexical Density: {0}%".format(summary.get("lexical_density", 0)),
        ]

    def _evidence_block(self, title, evidence, stance):
        lines = ["", title + ":"]
        filtered = [item for item in evidence if item.get("stance") == stance]
        if not filtered:
            lines.append("No {0} evidence found.".format(stance))
            return lines
        for index, item in enumerate(filtered[:3], start=1):
            lines.append("[{0}] {1}".format(index, item.get("sentence", "")))
            lines.append(
                "Source: {0} | Reliability: {1}/10 | Jaccard: {2}% | Cosine: {3}%".format(
                    item.get("domain", ""),
                    item.get("reliability", 0),
                    item.get("jaccard_similarity", 0),
                    item.get("cosine_similarity", 0),
                )
            )
        return lines

    def _diagnostics_block(self, retrieval):
        lines = ["", "SEARCH DIAGNOSTICS REPORT"]
        for page in retrieval.get("page_diagnostics", []):
            lines.append(
                "{0} | status={1} | accepted={2} | text_length={3} | quality={4}%".format(
                    page.get("domain", ""),
                    page.get("status_code"),
                    page.get("accepted"),
                    page.get("text_length", 0),
                    page.get("extraction_quality", 0),
                )
            )
            if page.get("rejection_reason"):
                lines.append("Reason: " + page.get("rejection_reason", ""))
        return lines

    def _keyword_scores(self, scores):
        if not scores:
            return "None"
        return ", ".join("{0}({1})".format(item["keyword"], item["score"]) for item in scores[:8])

    def _confidence_text(self, confidence):
        value = confidence.get("percentage")
        if value is None:
            return "Not available due to insufficient retrievable evidence"
        return str(value) + "%"

    def _join(self, values):
        values = [str(value) for value in values if value]
        return ", ".join(values) if values else "None"

    def _tuple_lines(self, rows):
        if not rows:
            return ["None"]
        return ["- {0}: {1}".format(row[0] or "Unknown", row[1]) for row in rows]
