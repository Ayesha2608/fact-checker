# Project Report

## Title

AI-Powered Evidence Analysis and Fact Verification Platform

## Objective

The objective is to build a transparent NLP-based fact verification platform that accepts natural-language questions or claims, runs a manual core NLP pipeline, evaluates the NLP pipeline offline, searches the live web, diagnoses retrieval quality, extracts evidence from multiple sources, identifies support and contradiction, computes confidence, evaluates verification quality, stores results, and displays citations.

## Methodology

The system uses only Python Standard Library modules. The assignment permits NLTK, but the submitted implementation keeps all NLP steps manual to make every rule explainable. It avoids machine learning frameworks and external NLP packages by combining regex tokenization, custom stopword removal, rule-based stemming and lemmatization, n-gram generation, heuristic POS tagging, named-entity heuristics, phrase chunking, claim frame extraction, negation-scope detection, claim category detection, sentiment analysis, HTML parsing, retrieval diagnostics, rule-based credibility scoring, keyword overlap, `difflib` similarity, and contradiction heuristics.

## Architecture

The application is modular:

- Query processing normalizes text, identifies intent, extracts named entities, estimates complexity, ranks keywords, detects category, estimates sentiment, and exposes a full NLP pipeline summary.
- The NLP pipeline performs sentence segmentation, tokenization, stopword removal, stemming, lemmatization, n-gram generation, heuristic POS tagging, NER, noun/verb phrase chunking, claim-structure analysis, negation scope detection, lexical diversity, lexical density, and readability proxies.
- Offline NLP evaluation loads labelled examples from `sample_data/nlp_evaluation.csv` and reports question type accuracy, category accuracy, keyword recall, entity recall, and per-example diagnostics.
- Live search uses `urllib.request` against a public HTML search endpoint and records the exact search URL, status code, response bytes, parsed results, and errors.
- Web extraction removes scripts, styles, and non-visible content with `html.parser`; each page receives status, accepted/rejected state, text length, and extraction quality.
- Evidence extraction ranks sentences by keyword frequency, token overlap, phrase hits, Jaccard similarity, cosine similarity, and string similarity.
- Credibility analysis assigns reliability from domain rules such as government, educational, research, news, blog, and social sources.
- Verification combines stance counts and weighted source reliability into verdicts.
- Confidence scoring reports a percentage and a label from Very Low to Very High.
- Hallucination detection compares generated explanations against extracted evidence.
- SQLite stores fact checks, sources, feedback, history, and cache records.
- Analytics screens summarize usage, categories, confidence trends, sources, keywords, and verdict statistics.

## Verdict Classes

- True
- Mostly True
- Mixed Evidence
- Mostly False
- False
- Insufficient Evidence

## Confidence Formula

The implemented formula follows the project requirement:

```text
Confidence = (Support Score / Total Evidence Score) * 100
```

Where support and contradiction scores are weighted by source reliability. Neutral evidence contributes partially to the denominator to lower confidence when evidence is weak or mixed. If no reliable evidence is retrieved, confidence is reported as Not Available instead of 0%.

## NLP Evaluation Metrics

The project includes transparent evaluation metrics suitable for viva and demonstration:

- Coverage Score: how much of the claim vocabulary appears in extracted evidence.
- Evidence Strength: average relevance and reliability of extracted evidence.
- Source Diversity: ratio of unique domains to total sources.
- Reliability Score: weighted source credibility from domain rules.
- Retrieval Quality: average extraction quality from downloaded pages.
- Overall Verification Quality: weighted aggregate of coverage, strength, diversity, reliability, and retrieval quality.
- Question Type Accuracy: how often the manual classifier matches labelled examples.
- Category Accuracy: how often topic classification matches labelled examples.
- Keyword Recall: percentage of expected important terms recovered by the pipeline.
- Entity Recall: percentage of expected entities found by the heuristic NER stage.

## Retrieval Failure Handling

If live search or scraping fails, the system returns `STATUS: INSUFFICIENT DATA`. It explains possible causes such as internet unavailability, blocked search endpoint, very recent query, unavailable pages, or insufficient extracted text. It also displays a Search Diagnostics Report with the exact search URL, response status, pages retrieved, pages rejected, extraction success rate, domains contacted, and parsed text lengths.

## Database Schema

The SQLite database creates these tables:

- `FactChecks`
- `Sources`
- `Feedback`
- `History`
- `Cache`

`Cache` is an optional bonus table for future result caching. The required tables are all present.

## Limitations

Because the project intentionally uses no external NLP or machine learning libraries, contradiction detection is heuristic and may miss subtle semantic contradictions, sarcasm, paraphrase, or complex context. Live web search also depends on public web page availability and network access. The system therefore uses `Insufficient Evidence` when retrieval fails and `Mixed Evidence` when evidence exists but remains balanced or weak. Offline evaluation is included so the core NLP pipeline can still be demonstrated without network access.

## Future Enhancements

- Add more domain-specific credibility rules.
- Use the existing cache table for cached search and page retrieval.
- Expand the labelled NLP evaluation corpus.
- Add a web interface using only standard-library `http.server`.
- Expand test coverage for search parsing edge cases.
