"""Evidence-based confidence scoring."""

from utils import clamp


class ConfidenceEngine:
    """Calculate confidence percentage and human-readable level."""

    LEVELS = [
        (85, "Very High"),
        (70, "High"),
        (45, "Medium"),
        (25, "Low"),
        (0, "Very Low"),
    ]

    def calculate(
        self,
        support_score,
        contradiction_score,
        neutral_score=0.0,
        source_credibility=0.0,
        similarity_score=0.0,
        support_sentence_count=0,
        contradiction_sentence_count=0,
    ):
        """Return confidence from weighted evidence, credibility, and similarity."""
        support_multiplier = 1.0
        support_multiplier += clamp(source_credibility, 0, 100) / 100 * 0.25
        support_multiplier += clamp(similarity_score, 0, 100) / 100 * 0.25
        support_multiplier += min(0.2, max(0, support_sentence_count) * 0.03)
        contradiction_multiplier = 1.0 + min(0.2, max(0, contradiction_sentence_count) * 0.03)
        adjusted_support = support_score * support_multiplier
        adjusted_contradiction = contradiction_score * contradiction_multiplier
        total = adjusted_support + adjusted_contradiction + neutral_score
        if total <= 0:
            percentage = 0
        else:
            percentage = int(round((adjusted_support / total) * 100))
        percentage = int(clamp(percentage, 0, 100))
        return {"percentage": percentage, "level": self.level(percentage)}

    def calculate_breakdown(self, evidence_strength, source_credibility, agreement, coverage):
        """Return confidence using the project formula and a transparent breakdown."""
        components = {
            "evidence_strength": int(clamp(evidence_strength, 0, 100)),
            "source_credibility": int(clamp(source_credibility, 0, 100)),
            "cross_source_agreement": int(clamp(agreement, 0, 100)),
            "claim_coverage": int(clamp(coverage, 0, 100)),
        }
        weighted = {
            "evidence_strength": round(components["evidence_strength"] * 0.30, 2),
            "source_credibility": round(components["source_credibility"] * 0.25, 2),
            "cross_source_agreement": round(components["cross_source_agreement"] * 0.25, 2),
            "claim_coverage": round(components["claim_coverage"] * 0.20, 2),
        }
        percentage = int(round(sum(weighted.values())))
        percentage = int(clamp(percentage, 0, 100))
        return {
            "percentage": percentage,
            "level": self.level(percentage),
            "breakdown": components,
            "weighted_breakdown": weighted,
            "formula": "0.30*evidence_strength + 0.25*source_credibility + 0.25*cross_source_agreement + 0.20*claim_coverage",
        }

    def level(self, percentage):
        """Return Very High, High, Medium, Low, or Very Low."""
        for threshold, label in self.LEVELS:
            if percentage >= threshold:
                return label
        return "Very Low"
