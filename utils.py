"""Shared utilities for the standard-library-only fact-checking project."""

import hashlib
import logging
import os
import pathlib
import re
import string
import urllib.parse
from datetime import datetime, timezone


APP_NAME = "FactCheckingChatbot"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def configure_logging(log_file="fact_checker.log"):
    """Configure application logging once."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def ensure_directory(path):
    """Create a directory if it does not already exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def normalize_space(text):
    """Collapse repeated whitespace into single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def clean_text(text):
    """Normalize common noisy characters in downloaded text."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return normalize_space(text)


def tokenize_words(text):
    """Tokenize text into lowercase alphanumeric terms using regex only."""
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text or "")]


def remove_punctuation_edges(text):
    """Strip punctuation from the beginning and end of a token."""
    return (text or "").strip(string.punctuation + " ")


def split_sentences(text, min_length=8, max_length=700):
    """Split a document into readable sentences with a regex heuristic."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", cleaned)
    sentences = []
    for piece in pieces:
        piece = normalize_space(piece)
        if min_length <= len(piece) <= max_length:
            sentences.append(piece)
    return sentences


def domain_from_url(url):
    """Extract a lowercase domain from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.split(":")[0]


def url_hash(value):
    """Return a stable hash suitable for cache keys."""
    return hashlib.sha256((value or "").encode("utf-8", errors="ignore")).hexdigest()


def current_timestamp():
    """Return a compact ISO timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def display_timestamp():
    """Return a human-readable timestamp for reports."""
    return datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


def format_claim_id(number, timestamp=None):
    """Format a database ID as a professional claim identifier."""
    year = (timestamp or datetime.now()).year
    return "FC-{0}-{1:06d}".format(year, int(number or 0))


def clamp(value, low, high):
    """Clamp a number to an inclusive range."""
    return max(low, min(high, value))


def safe_float(value, default=0.0):
    """Convert a value to float without raising."""
    try:
        return float(value)
    except Exception:
        return default


def project_path(*parts):
    """Build a path relative to the current working directory."""
    return os.path.join(os.getcwd(), *parts)
