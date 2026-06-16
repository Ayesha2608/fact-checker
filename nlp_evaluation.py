"""Offline NLP evaluation helpers for final-year project demonstration."""

import csv
import time
from collections import Counter
from pathlib import Path

from query_processor import QueryProcessor
from verifier import FactCheckingEngine


class NLPEvaluator:
    """Evaluate the manual NLP pipeline against a small labelled corpus."""

    DEFAULT_DATASET = Path("sample_data") / "nlp_evaluation.csv"

    def __init__(self, dataset_path=None):
        self.dataset_path = Path(dataset_path) if dataset_path else self.DEFAULT_DATASET
        self.processor = QueryProcessor()

    def load_dataset(self):
        """Load labelled examples from CSV."""
        if not self.dataset_path.exists():
            return []
        with self.dataset_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def evaluate(self, rows=None):
        """Return transparent accuracy-style metrics for NLP components."""
        examples = rows if rows is not None else self.load_dataset()
        if not examples:
            return {
                "examples": 0,
                "question_type_accuracy": 0,
                "category_accuracy": 0,
                "keyword_recall": 0,
                "entity_recall": 0,
                "details": [],
            }
        question_hits = 0
        category_hits = 0
        keyword_scores = []
        entity_scores = []
        details = []
        for row in examples:
            result = self.processor.process(row.get("text", ""))
            question_ok = result["question_type"] == row.get("expected_question_type", "")
            category_ok = result["category"] == row.get("expected_category", "")
            keyword_recall = self._recall(self._split(row.get("expected_keywords", "")), result.get("tokens", []))
            detected_entities = self._flatten_entities(result.get("entities", {}))
            entity_recall = self._recall(self._split(row.get("expected_entities", "")), detected_entities)
            question_hits += 1 if question_ok else 0
            category_hits += 1 if category_ok else 0
            keyword_scores.append(keyword_recall)
            entity_scores.append(entity_recall)
            details.append(
                {
                    "text": row.get("text", ""),
                    "expected_question_type": row.get("expected_question_type", ""),
                    "predicted_question_type": result["question_type"],
                    "expected_category": row.get("expected_category", ""),
                    "predicted_category": result["category"],
                    "keyword_recall": keyword_recall,
                    "entity_recall": entity_recall,
                }
            )
        total = len(examples)
        return {
            "examples": total,
            "question_type_accuracy": self._percentage(question_hits, total),
            "category_accuracy": self._percentage(category_hits, total),
            "keyword_recall": int(round(sum(keyword_scores) / max(1, total))),
            "entity_recall": int(round(sum(entity_scores) / max(1, total))),
            "details": details,
        }

    def demo_pipeline(self, text):
        """Return the full processed output for a single text sample."""
        return self.processor.process(text)

    def evaluate_verification(self, rows=None, engine=None):
        """Evaluate end-to-end verification on a labelled claim dataset."""
        examples = rows if rows is not None else []
        engine = engine or FactCheckingEngine(database_path=":memory:")
        if not examples:
            return {
                "examples": 0,
                "retrieval_precision": 0,
                "retrieval_recall": 0,
                "evidence_f1": 0,
                "stance_accuracy": 0,
                "verdict_accuracy": 0,
                "confidence_calibration": 0,
                "average_response_time": 0,
                "source_extraction_success_rate": 0,
                "confusion_matrix": {},
                "error_analysis": [],
            }
        verdict_hits = 0
        stance_hits = 0
        precision_scores = []
        recall_scores = []
        f1_scores = []
        calibration_scores = []
        response_times = []
        extraction_scores = []
        confusion = Counter()
        errors = []
        for row in examples:
            started = time.time()
            result = engine.verify_claim(row.get("text", ""), mode=row.get("mode", "general"))
            response_times.append(round(time.time() - started, 3))
            expected_verdict = row.get("expected_verdict", "")
            expected_stance = row.get("expected_stance", "")
            predicted_verdict = result.get("verdict", "")
            confusion[(expected_verdict, predicted_verdict)] += 1
            if expected_verdict and expected_verdict == predicted_verdict:
                verdict_hits += 1
            stances = [item.get("stance") for item in result.get("evidence", [])]
            majority_stance = Counter(stances).most_common(1)[0][0] if stances else "neutral"
            if expected_stance and expected_stance == majority_stance:
                stance_hits += 1
            relevant = len([item for item in result.get("evidence", []) if item.get("stance") in {"support", "contradict"}])
            retrieved = max(1, len(result.get("evidence", [])))
            precision = relevant / retrieved
            recall = 1.0 if relevant > 0 else 0.0
            f1 = (2 * precision * recall / max(0.0001, precision + recall)) if relevant else 0.0
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            confidence = result.get("confidence", {}).get("percentage")
            if confidence is not None and expected_verdict:
                correct = expected_verdict == predicted_verdict
                calibration_scores.append(100 - abs((100 if correct else 0) - confidence))
            extraction_scores.append(result.get("retrieval_analysis", {}).get("extraction_success_rate", 0))
            if expected_verdict and expected_verdict != predicted_verdict:
                errors.append(
                    {
                        "text": row.get("text", ""),
                        "expected": expected_verdict,
                        "predicted": predicted_verdict,
                        "evidence_count": len(result.get("evidence", [])),
                        "sources": result.get("retrieval_analysis", {}).get("sources_searched", 0),
                    }
                )
        total = len(examples)
        return {
            "examples": total,
            "retrieval_precision": self._average_percentage(precision_scores),
            "retrieval_recall": self._average_percentage(recall_scores),
            "evidence_f1": self._average_percentage(f1_scores),
            "stance_accuracy": self._percentage(stance_hits, total),
            "verdict_accuracy": self._percentage(verdict_hits, total),
            "confidence_calibration": int(round(sum(calibration_scores) / max(1, len(calibration_scores)))),
            "average_response_time": round(sum(response_times) / max(1, total), 3),
            "source_extraction_success_rate": int(round(sum(extraction_scores) / max(1, total))),
            "confusion_matrix": {
                expected + " -> " + predicted: count
                for (expected, predicted), count in confusion.items()
            },
            "error_analysis": errors,
        }

    def _split(self, value):
        return [item.strip().lower() for item in (value or "").split("|") if item.strip()]

    def _flatten_entities(self, entities):
        values = []
        for group in entities.values():
            values.extend(str(item).lower() for item in group)
        return values

    def _recall(self, expected, actual):
        if not expected:
            return 100
        actual_set = set(actual)
        hits = len([item for item in expected if item in actual_set])
        return self._percentage(hits, len(expected))

    def _percentage(self, numerator, denominator):
        return int(round((numerator / max(1, denominator)) * 100))

    def _average_percentage(self, values):
        return int(round((sum(values) / max(1, len(values))) * 100))
