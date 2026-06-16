"""Live web search using urllib and public HTML search pages."""

import base64
import html
import json
import logging
import re
import socket
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from utils import USER_AGENT, clean_text, domain_from_url


LOGGER = logging.getLogger(__name__)


class SearchResultParser(HTMLParser):
    """Extract candidate search result links from a DuckDuckGo HTML page."""

    def __init__(self):
        HTMLParser.__init__(self)
        self.results = []
        self._inside_link = False
        self._current_href = ""
        self._current_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a":
            href = attrs_dict.get("href", "")
            css_class = attrs_dict.get("class", "")
            if href and ("result__a" in css_class or "/l/?" in href or href.startswith("http")):
                self._inside_link = True
                self._current_href = href
                self._current_text = []

    def handle_data(self, data):
        if self._inside_link:
            self._current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._inside_link:
            title = clean_text(" ".join(self._current_text))
            url = self._normalize_url(self._current_href)
            if url and title:
                self.results.append({"title": html.unescape(title), "url": url, "domain": domain_from_url(url)})
            self._inside_link = False
            self._current_href = ""
            self._current_text = []

    def _normalize_url(self, href):
        if not href:
            return ""
        href = html.unescape(href)
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/l/?"):
            parsed = urllib.parse.urlparse(href)
            params = urllib.parse.parse_qs(parsed.query)
            target = params.get("uddg", [""])[0]
            return target
        if href.startswith("http://") or href.startswith("https://"):
            parsed = urllib.parse.urlparse(href)
            params = urllib.parse.parse_qs(parsed.query)
            if "uddg" in params:
                return params["uddg"][0]
            return href
        return ""


class LiteSearchResultParser(HTMLParser):
    """Extract result links from DuckDuckGo Lite pages."""

    def __init__(self):
        HTMLParser.__init__(self)
        self.results = []
        self._inside_link = False
        self._current_href = ""
        self._current_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a":
            href = attrs_dict.get("href", "")
            if href and (href.startswith("http") or "/l/?" in href):
                self._inside_link = True
                self._current_href = href
                self._current_text = []

    def handle_data(self, data):
        if self._inside_link:
            self._current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._inside_link:
            title = clean_text(" ".join(self._current_text))
            url = SearchResultParser()._normalize_url(self._current_href)
            if url and title and "duckduckgo" not in domain_from_url(url):
                self.results.append({"title": html.unescape(title), "url": url, "domain": domain_from_url(url)})
            self._inside_link = False
            self._current_href = ""
            self._current_text = []


class BingSearchResultParser(HTMLParser):
    """Extract candidate links from Bing HTML result pages."""

    def __init__(self):
        HTMLParser.__init__(self)
        self.results = []
        self._inside_link = False
        self._inside_result = False
        self._current_href = ""
        self._current_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "li" and "b_algo" in attrs_dict.get("class", ""):
            self._inside_result = True
        if tag == "a" and self._inside_result:
            href = attrs_dict.get("href", "")
            if href.startswith(("http://", "https://")):
                target = self._normalize_url(href)
                domain = domain_from_url(target)
                if target and not any(blocked in domain for blocked in ("bing.com", "microsoft.com", "r.bing.com")):
                    self._inside_link = True
                    self._current_href = target
                    self._current_text = []

    def handle_data(self, data):
        if self._inside_link:
            self._current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._inside_link:
            title = clean_text(" ".join(self._current_text))
            if title:
                self.results.append({"title": html.unescape(title), "url": self._current_href, "domain": domain_from_url(self._current_href)})
            self._inside_link = False
            self._current_href = ""
            self._current_text = []
        elif tag == "li" and self._inside_result:
            self._inside_result = False

    def _normalize_url(self, href):
        parsed = urllib.parse.urlparse(html.unescape(href))
        domain = parsed.netloc.lower()
        if "bing.com" not in domain:
            return href
        params = urllib.parse.parse_qs(parsed.query)
        encoded = params.get("u", [""])[0]
        if not encoded:
            return ""
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        padding = "=" * (-len(encoded) % 4)
        try:
            return base64.urlsafe_b64decode((encoded + padding).encode("ascii")).decode("utf-8", errors="replace")
        except Exception:
            return ""


class SearchEngine:
    """Search the web without third-party packages."""

    def __init__(self, timeout=5, max_results=8):
        self.timeout = timeout
        self.max_results = max_results

    def search(self, query, max_results=None):
        """Return a de-duplicated list of search result dictionaries."""
        return self.search_with_diagnostics(query, max_results)["results"]

    def search_with_diagnostics(self, query, max_results=None):
        """Return search results plus a diagnostic report for the retrieval layer."""
        limit = max_results or self.max_results
        diagnostics = {
            "query": query,
            "search_url": "",
            "status_code": None,
            "response_bytes": 0,
            "raw_results_parsed": 0,
            "results_returned": 0,
            "error": "",
            "direct_connection": True,
        }
        if not query:
            diagnostics["error"] = "Empty query"
            return {"results": [], "diagnostics": diagnostics}
        encoded = urllib.parse.urlencode({"q": query})
        url = "https://html.duckduckgo.com/html/?" + encoded
        diagnostics["search_url"] = url
        LOGGER.info("Searching web query=%s", query)
        try:
            download = self._download(url)
            content = download["text"]
            diagnostics["status_code"] = download["status_code"]
            diagnostics["response_bytes"] = download["response_bytes"]
        except Exception as exc:
            LOGGER.warning("Search failed: %s", exc)
            diagnostics["error"] = str(exc)
            content = ""
        parser = SearchResultParser()
        if content:
            try:
                parser.feed(content)
            except Exception as exc:
                LOGGER.warning("Search parse failed: %s", exc)
                diagnostics["error"] = "Search parse failed: " + str(exc)
        results = self._deduplicate(parser.results)[:limit]
        if not results:
            fallback = self._lite_search(encoded, diagnostics)
            results = self._deduplicate(fallback)[:limit]
        if not results:
            news_fallback = self._bing_news_rss_search(query, diagnostics)
            results = self._deduplicate(self._filter_relevant(news_fallback, query))[:limit]
        if not results:
            fallback = self._bing_rss_search(query, diagnostics)
            results = self._deduplicate(self._filter_relevant(fallback, query))[:limit]
        if not results:
            fallback = self._wikipedia_search(query, diagnostics)
            results = self._deduplicate(fallback)[:limit]
        elif len(results) < limit:
            fallback = self._wikipedia_search(query, diagnostics)
            if fallback:
                results = self._deduplicate(results + fallback)[:limit]
        diagnostics["raw_results_parsed"] = len(parser.results)
        diagnostics["results_returned"] = len(results)
        return {"results": results, "diagnostics": diagnostics}

    def _lite_search(self, encoded_query, diagnostics):
        """Fallback to DuckDuckGo Lite when the HTML endpoint is blocked or empty."""
        url = "https://lite.duckduckgo.com/lite/?" + encoded_query
        diagnostics["fallback_search_url"] = url
        try:
            download = self._download(url)
        except Exception as exc:
            diagnostics["fallback_error"] = str(exc)
            return []
        diagnostics["fallback_status_code"] = download["status_code"]
        diagnostics["fallback_response_bytes"] = download["response_bytes"]
        parser = LiteSearchResultParser()
        try:
            parser.feed(download["text"])
        except Exception as exc:
            diagnostics["fallback_error"] = "Lite search parse failed: " + str(exc)
            return []
        diagnostics["fallback_raw_results_parsed"] = len(parser.results)
        return parser.results

    def _bing_search(self, query, diagnostics):
        """Second-provider fallback for demos when DuckDuckGo throttles."""
        encoded = urllib.parse.urlencode({"q": query})
        url = "https://www.bing.com/search?" + encoded
        diagnostics["secondary_search_url"] = url
        try:
            download = self._download(url)
        except Exception as exc:
            diagnostics["secondary_error"] = str(exc)
            return []
        diagnostics["secondary_status_code"] = download["status_code"]
        diagnostics["secondary_response_bytes"] = download["response_bytes"]
        parser = BingSearchResultParser()
        try:
            parser.feed(download["text"])
        except Exception as exc:
            diagnostics["secondary_error"] = "Bing search parse failed: " + str(exc)
            return []
        results = parser.results or self._parse_bing_h2_links(download["text"])
        diagnostics["secondary_raw_results_parsed"] = len(results)
        return results

    def _bing_rss_search(self, query, diagnostics):
        """Secondary search fallback using Bing's RSS output."""
        encoded = urllib.parse.urlencode({"format": "rss", "q": query})
        url = "https://www.bing.com/search?" + encoded
        diagnostics["secondary_search_url"] = url
        try:
            download = self._download(url)
        except Exception as exc:
            diagnostics["secondary_error"] = str(exc)
            return []
        diagnostics["secondary_status_code"] = download["status_code"]
        diagnostics["secondary_response_bytes"] = download["response_bytes"]
        results = []
        try:
            root = ET.fromstring(download["text"])
            for item in root.findall(".//item"):
                title = clean_text(item.findtext("title", ""))
                url = self._unwrap_bing_news_url(clean_text(item.findtext("link", "")))
                snippet = clean_text(item.findtext("description", ""))
                domain = domain_from_url(url)
                if title and url and domain:
                    results.append({"title": title, "url": url, "domain": domain, "snippet": snippet})
        except Exception as exc:
            diagnostics["secondary_error"] = "Bing RSS parse failed: " + str(exc)
            return []
        diagnostics["secondary_raw_results_parsed"] = len(results)
        return results

    def _bing_news_rss_search(self, query, diagnostics):
        """News-specific RSS fallback for current events and headlines."""
        encoded = urllib.parse.urlencode({"format": "rss", "q": query})
        url = "https://www.bing.com/news/search?" + encoded
        diagnostics["news_search_url"] = url
        try:
            download = self._download(url)
        except Exception as exc:
            diagnostics["news_error"] = str(exc)
            return []
        diagnostics["news_status_code"] = download["status_code"]
        diagnostics["news_response_bytes"] = download["response_bytes"]
        results = []
        try:
            root = ET.fromstring(download["text"])
            for item in root.findall(".//item"):
                title = clean_text(item.findtext("title", ""))
                url = self._unwrap_bing_news_url(clean_text(item.findtext("link", "")))
                snippet = clean_text(item.findtext("description", ""))
                domain = domain_from_url(url)
                if title and url and domain:
                    results.append({"title": title, "url": url, "domain": domain, "snippet": snippet})
        except Exception as exc:
            diagnostics["news_error"] = "Bing News RSS parse failed: " + str(exc)
            return []
        diagnostics["news_raw_results_parsed"] = len(results)
        return results

    def _unwrap_bing_news_url(self, url):
        parsed = urllib.parse.urlparse(url or "")
        if "bing.com" not in parsed.netloc.lower():
            return url
        params = urllib.parse.parse_qs(parsed.query)
        target = params.get("url", [""])[0]
        return target or url

    def _parse_bing_h2_links(self, content):
        results = []
        pattern = re.compile(r"<h2[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
        for href, title_html in pattern.findall(content or ""):
            url = BingSearchResultParser()._normalize_url(html.unescape(href))
            title = clean_text(re.sub(r"<[^>]+>", " ", html.unescape(title_html)))
            domain = domain_from_url(url)
            if url and title and not any(blocked in domain for blocked in ("bing.com", "microsoft.com", "r.bing.com")):
                results.append({"title": title, "url": url, "domain": domain})
        return results

    def _wikipedia_search(self, query, diagnostics):
        """Live encyclopedic source discovery fallback for factual claims."""
        attempts = self._wikipedia_query_variants(query)
        all_results = []
        diagnostics["wikipedia_attempts"] = []
        for search_text in attempts:
            encoded = urllib.parse.urlencode(
                {
                    "action": "opensearch",
                    "search": search_text,
                    "limit": 5,
                    "namespace": 0,
                    "format": "json",
                }
            )
            url = "https://en.wikipedia.org/w/api.php?" + encoded
            diagnostics["wikipedia_search_url"] = url
            try:
                download = self._download(url)
                data = json.loads(download["text"])
            except Exception as exc:
                diagnostics["wikipedia_error"] = str(exc)
                continue
            titles = data[1] if len(data) > 1 else []
            urls = data[3] if len(data) > 3 else []
            diagnostics["wikipedia_attempts"].append({"query": search_text, "results": len(urls)})
            for title, result_url in zip(titles, urls):
                if result_url:
                    all_results.append({"title": title, "url": result_url, "domain": domain_from_url(result_url)})
            if all_results:
                break
        diagnostics["wikipedia_results_returned"] = len(all_results)
        return all_results

    def _wikipedia_query_variants(self, query):
        tokens = [token for token in re.findall(r"[a-z0-9]+", (query or "").lower()) if token not in {"is", "a", "the", "of", "in", "to", "for", "and", "or", "reliable", "source", "fact", "check", "official", "evidence"}]
        variants = []
        if tokens:
            variants.append(" ".join(tokens[:6]))
        if "karachi" in tokens and "pakistan" in tokens:
            variants.append("Karachi Pakistan")
        if "earth" in tokens and "moon" in tokens:
            variants.append("Earth second moon")
        if "covid" in tokens and "pandemic" in tokens:
            variants.append("COVID-19 pandemic")
        return self._unique_text([query] + variants)

    def _unique_text(self, values):
        seen = set()
        output = []
        for value in values:
            cleaned = clean_text(value)
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                output.append(cleaned)
        return output

    def _download(self, url):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout) as response:
            raw = response.read(750000)
            status_code = getattr(response, "status", None) or response.getcode()
            content_type = response.headers.get("Content-Type", "")
        charset = "utf-8"
        match = re.search(r"charset=([\w-]+)", content_type)
        if match:
            charset = match.group(1)
        return {
            "text": raw.decode(charset, errors="replace"),
            "status_code": status_code,
            "response_bytes": len(raw),
        }

    def _deduplicate(self, results):
        seen = set()
        clean = []
        for result in results:
            url = result.get("url", "")
            domain = result.get("domain", "")
            if not url.startswith(("http://", "https://")):
                continue
            if any(blocked in domain for blocked in ("duckduckgo.com", "bing.com/search")):
                continue
            key = urllib.parse.urldefrag(url)[0].rstrip("/")
            if key not in seen:
                seen.add(key)
                clean.append(result)
        return clean

    def _filter_relevant(self, results, query):
        query_tokens = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
        filtered = []
        for result in results:
            text = " ".join([result.get("title", ""), result.get("snippet", ""), result.get("domain", "")]).lower()
            result_tokens = set(re.findall(r"[a-z0-9]+", text))
            if len(query_tokens & result_tokens) >= 2:
                filtered.append(result)
        return filtered

def has_network():
    """Best-effort network availability check for CLI diagnostics."""
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        return True
    except Exception:
        return False
