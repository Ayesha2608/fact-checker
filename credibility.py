"""Rule-based source credibility scoring."""

import logging

from utils import domain_from_url


LOGGER = logging.getLogger(__name__)


class CredibilityAnalyzer:
    """Assign reliability scores from 1 to 10 using transparent rules."""

    RESEARCH_TERMS = {
        "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "nih.gov",
        "who.int", "nature.com", "science.org", "springer.com", "sciencedirect.com",
        "jstor.org", "ieee.org", "acm.org", "doi.org", "plos.org",
    }
    NEWS_TERMS = {
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "theguardian.com",
        "nytimes.com", "washingtonpost.com", "aljazeera.com", "cnn.com", "nbcnews.com",
        "cbsnews.com", "abcnews.go.com", "forbes.com", "bloomberg.com",
    }
    FACT_CHECK_TERMS = {
        "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
        "afp.com", "reuters.com/fact-check", "apnews.com/hub/ap-fact-check",
    }
    LOW_TRUST_TERMS = {
        "blogspot.", "wordpress.", "medium.com", "substack.com", "facebook.com",
        "x.com", "twitter.com", "tiktok.com", "instagram.com", "reddit.com",
    }

    def score(self, url):
        """Return a reliability result with score and reason."""
        domain = domain_from_url(url)
        score = 5
        reason = "General web source"
        tier = "Tier 3"
        if domain.endswith(".gov") or ".gov." in domain:
            score, reason, tier = 10, "Government domain", "Tier 1"
        elif domain.endswith(".edu") or ".edu." in domain or ".ac." in domain:
            score, reason, tier = 9, "Educational domain", "Tier 1"
        elif self._contains(domain, self.FACT_CHECK_TERMS):
            score, reason, tier = 9, "Dedicated fact-checking source", "Tier 1"
        elif self._contains(domain, self.RESEARCH_TERMS):
            score, reason, tier = 9, "Research or scientific source", "Tier 1"
        elif self._contains(domain, self.NEWS_TERMS):
            score, reason, tier = 8, "Established news source", "Tier 2"
        elif self._contains(domain, self.LOW_TRUST_TERMS):
            score, reason, tier = 3, "User-generated or blog-style source", "Tier 4"
        elif domain.endswith(".org"):
            score, reason, tier = 6, "Organization domain", "Tier 3"
        return {
            "domain": domain,
            "reliability": score,
            "normalized": round(score / 10, 2),
            "tier": tier,
            "reason": reason,
        }

    def _contains(self, domain, terms):
        for term in terms:
            if term in domain:
                return True
        return False
