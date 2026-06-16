"""Web page download and visible-text extraction using html.parser."""

import html
import logging
import re
import urllib.request
from html.parser import HTMLParser

from utils import USER_AGENT, clean_text


LOGGER = logging.getLogger(__name__)


class VisibleTextParser(HTMLParser):
    """Extract title and visible text while ignoring scripts and styles."""

    BLOCKED_TAGS = {"script", "style", "noscript", "svg", "canvas", "iframe"}

    def __init__(self):
        HTMLParser.__init__(self)
        self.text_parts = []
        self.title_parts = []
        self._blocked_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCKED_TAGS:
            self._blocked_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self.BLOCKED_TAGS and self._blocked_depth > 0:
            self._blocked_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._blocked_depth:
            return
        text = clean_text(html.unescape(data))
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        else:
            self.text_parts.append(text)

    def result(self):
        """Return title and visible text."""
        return {
            "title": clean_text(" ".join(self.title_parts)),
            "text": clean_text(" ".join(self.text_parts)),
        }


class WebScraper:
    """Download pages and extract readable text."""

    def __init__(self, timeout=10, max_bytes=1200000):
        self.timeout = timeout
        self.max_bytes = max_bytes

    def fetch(self, url):
        """Fetch and parse a URL into a page dictionary."""
        try:
            download = self._download(url)
            parsed = self.extract_text(download["text"])
            parsed["url"] = url
            parsed["error"] = ""
            parsed["status_code"] = download["status_code"]
            parsed["content_type"] = download["content_type"]
            parsed["response_bytes"] = download["response_bytes"]
            parsed["text_length"] = len(parsed.get("text", ""))
            parsed["retrieved"] = parsed["status_code"] and 200 <= int(parsed["status_code"]) < 400
            parsed["accepted"] = parsed["retrieved"] and parsed["text_length"] >= 200
            parsed["rejection_reason"] = "" if parsed["accepted"] else self._rejection_reason(parsed)
            parsed["extraction_quality"] = self._quality(parsed)
            return parsed
        except Exception as exc:
            LOGGER.warning("Failed to fetch %s: %s", url, exc)
            return {
                "url": url,
                "title": "",
                "text": "",
                "error": str(exc),
                "status_code": None,
                "content_type": "",
                "response_bytes": 0,
                "text_length": 0,
                "retrieved": False,
                "accepted": False,
                "rejection_reason": str(exc),
                "extraction_quality": 0,
            }

    def _download(self, url):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout) as response:
            raw = response.read(self.max_bytes)
            status_code = getattr(response, "status", None) or response.getcode()
            content_type = response.headers.get("Content-Type", "")
        charset = "utf-8"
        match = re.search(r"charset=([\w-]+)", content_type)
        if match:
            charset = match.group(1)
        return {
            "text": raw.decode(charset, errors="replace"),
            "status_code": status_code,
            "content_type": content_type,
            "response_bytes": len(raw),
        }

    def extract_text(self, html_text):
        """Extract visible text from an HTML string."""
        parser = VisibleTextParser()
        parser.feed(html_text or "")
        return parser.result()

    def _quality(self, parsed):
        """Estimate extraction quality from text size and successful retrieval."""
        if not parsed.get("retrieved"):
            return 0
        text_length = parsed.get("text_length", 0)
        if text_length >= 3000:
            return 95
        if text_length >= 1200:
            return 82
        if text_length >= 500:
            return 65
        if text_length >= 200:
            return 45
        return 20

    def _rejection_reason(self, parsed):
        if not parsed.get("retrieved"):
            return parsed.get("error") or "HTTP request did not complete successfully"
        if parsed.get("text_length", 0) < 200:
            return "Extracted text was too short for reliable NLP evidence analysis"
        return ""
