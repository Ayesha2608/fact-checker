# VeriNova Product Design Specification

## Product Positioning

VeriNova is an AI fact-checking assistant for live evidence retrieval. The interface is designed as an evidence intelligence dashboard rather than a basic chatbot, so evaluators can immediately see how a claim moves through search, retrieval, NLP analysis, verification, and explanation.

## User Flow

1. User enters a claim or news headline.
2. User chooses General or News mode.
3. VeriNova decomposes the claim into entities, anchors, and claim structure.
4. The backend searches live sources and retrieves page text.
5. Evidence is classified as supporting, contradicting, or neutral.
6. The dashboard displays verdict, confidence, source reliability, evidence agreement, source cards, and reasoning.

## Dashboard Wireframe

```text
Header
  Brand: VeriNova
  Backend status

Left Claim Console
  Claim textarea
  General / News mode
  Verify button
  Demo prompts
  Live summary: sources analyzed, evidence coverage, avg credibility

Right Evidence Workbench
  Verdict card + confidence gauge
  Pipeline: Claim -> Search -> Retrieve -> Analyze -> Verify -> Explain
  Analytics cards
  AI explanation
  Claim decomposition
  Source reliability chart
  Evidence agreement chart
  Supporting evidence panel
  Contradicting evidence panel
  Source cards
  Retrieval diagnostics
```

## Component Hierarchy

- `ProductHeader`
- `ClaimConsole`
- `VerdictCard`
- `ConfidenceGauge`
- `PipelineStepper`
- `AnalyticsCards`
- `AIExplanation`
- `ClaimDecomposition`
- `ReliabilityDistribution`
- `EvidenceAgreement`
- `EvidencePanels`
- `SourceCards`
- `RetrievalDiagnostics`

## Color Palette

- Void background: `#050713`
- Deep panel: `rgba(13, 18, 38, 0.74)`
- Primary text: `#f7f8ff`
- Muted text: `#aab4d6`
- Cyan accent: `#62e6ff`
- Mint support: `#52f3a9`
- Amber neutral: `#ffd36e`
- Rose highlight: `#ff6aa8`
- Red contradiction: `#ff6578`
- Violet intelligence accent: `#9b8cff`

## Suggested Charts

- Confidence gauge: conic CSS gauge.
- Evidence agreement: stacked stance strip.
- Source credibility distribution: high, medium, low bar chart.
- Search coverage: numeric card from NLP/evidence coverage score.

## Evaluation Metrics Shown

The UI avoids fake training metrics such as 100% accuracy or recall. It displays runtime verification metrics:

- Sources analyzed
- Supporting sources
- Contradicting sources
- Average source credibility
- Evidence coverage
- Evidence agreement
- Search and extraction diagnostics

## Visual Language

The design uses glassmorphism, subtle starfield animation, restrained gradients, soft borders, and dense dashboard cards to communicate an AI research product. The interface is responsive and suitable for a university Final Year Project showcase, portfolio demo, and industry-facing presentation.
