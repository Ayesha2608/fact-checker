# AI-Powered Evidence Analysis and Fact Verification Platform

This final year NLP project is a professional command-line and browser-based fact verification platform built with Python Standard Library modules only. The assignment permits NLTK, but this implementation keeps the NLP pipeline fully manual for transparency and viva defense. It performs rule-based NLP analysis, offline NLP evaluation, live search diagnostics, page retrieval assessment, source credibility scoring, Jaccard/Cosine evidence ranking, contradiction detection, confidence scoring, risk analysis, quality scoring, citations, SQLite history, feedback storage, analytics dashboards, and CSV/text reporting.

## Dependency Policy

No third-party package is required or imported. Python Standard Library is used throughout. NLTK is allowed by the assignment but intentionally not required here. Do not install `requests`, `beautifulsoup`, `numpy`, `pandas`, `spacy`, `torch`, `tensorflow`, `scikit-learn`, `transformers`, or any other pip package for this project.

## Run

```bash
python main.py
```

## Web App Deployment

To run the browser UI locally:

```bash
python web_server.py
```

Then open:

```text
http://127.0.0.1:8001
```

For deployment on a hosting platform:

1. Use `web_server.py` as the start command.
2. Make sure the platform sets a `PORT` environment variable.
3. The server now binds to `0.0.0.0` and uses `PORT` automatically.
4. Keep the project directory writable so SQLite can store `fact_checker.db`.
5. Ensure outbound internet access is allowed, because live fact-checking depends on search and page retrieval.

Example start command:

```bash
python web_server.py
```

If your platform requires a host and port explicitly, it will still work because the server reads `PORT` from the environment.

The live search feature requires internet access. If retrieval fails, the platform does not show a misleading `0%` confidence score. It reports `INSUFFICIENT DATA` and displays the search URL, status code, domains contacted, page rejection reasons, extraction success rate, and text lengths.

## Menu

1. Verify Claim / Ask Question
2. Verify News
3. View History
4. Search History
5. Feedback
6. Analytics Dashboard
7. Export History CSV
8. Export Latest Full Report
9. NLP Pipeline Demo / Offline Evaluation
10. Exit

## Project Files

- `main.py` - interactive CLI
- `query_processor.py` - tokenization, stopword removal, keyword extraction, intent detection
- `nlp_pipeline.py` - manual NLP pipeline: normalization, sentence splitting, tokenization, stopword removal, stemming, lemmatization, n-grams, POS tags, NER, chunks, claim frames, negation scope, lexical metrics
- `nlp_evaluation.py` - offline labelled-corpus evaluation for NLP pipeline accuracy and recall
- `search_engine.py` - public web search using `urllib.request` with diagnostics
- `web_scraper.py` - HTML visible-text extraction and retrieval quality checks
- `credibility.py` - source reliability scoring
- `evidence_extractor.py` - sentence ranking and relevance scoring
- `contradiction_detector.py` - rule-based contradiction detection
- `verifier.py` - claim verification workflow
- `confidence.py` - confidence percentage and confidence label
- `hallucination_detector.py` - unsupported-statement checks
- `database.py` - SQLite schema and persistence
- `reporting.py` - dashboard-style text reports, analytics, and CSV reports
- `utils.py` - shared utilities
- `tests/test_core.py` - offline unit tests
- `sample_data/nlp_evaluation.csv` - labelled NLP examples for offline evaluation

## Professional Analysis Output

Each verification report includes:

- Claim ID and timestamp
- Query analysis with detected keywords, named entities, category, complexity, and sentiment
- NLP pipeline artifacts including lemmas, n-grams, POS tags, phrase chunks, claim structure, negation scope, lexical diversity, and lexical density
- Retrieval diagnostics with exact search URL, status code, retrieved domains, failed sources, accepted pages, rejected pages, and extraction quality
- Source reliability analysis by government, educational, research, news, and other sources
- Supporting, contradicting, and neutral evidence
- Verdict, confidence, and reasoning engine explanation
- Risk analysis for manipulation, bias, evidence coverage, and ambiguity
- Quality metrics for coverage, evidence strength, diversity, reliability, retrieval quality, and overall verification quality
- Citations and optional fake-news heuristic analysis

## Analytics Dashboard

The platform tracks:

- Total fact checks
- Average confidence
- Average quality score
- Most common categories
- Top sources
- Most frequent keywords
- Verification statistics
- Daily usage

## Offline NLP Evaluation

The project includes a small labelled corpus in `sample_data/nlp_evaluation.csv`. Menu option 9 runs offline evaluation for:

- question type accuracy
- category accuracy
- keyword recall
- entity recall
- per-example diagnostics

This is useful for viva because the NLP pipeline can be demonstrated even when the internet is unavailable.

## Tests

```bash
python -m unittest discover tests
```

## Notes

Search result pages can change over time, and public search endpoints may throttle automated traffic. The code handles failures gracefully and returns `Insufficient Evidence` when evidence cannot be retrieved.
