import {
  formatLastSubmittedClaim,
  normalizeClaimText,
  shouldAutoVerifyOnInput,
  shouldClearStaleResult,
  shouldSubmitOnEnter,
} from "./flow.mjs";

const API_BASE_URL = ["127.0.0.1", "localhost"].includes(window.location.hostname)
  ? ""
  : "https://verinova.onrender.com";

const state = {
  mode: "general",
  latestResult: null,
  lastSubmittedClaim: "",
  autoVerifyTimer: null,
  verificationToken: 0,
};

const claimInput = document.querySelector("#claimInput");
const verifyBtn = document.querySelector("#verifyBtn");
const clearBtn = document.querySelector("#clearBtn");
const serverStatus = document.querySelector("#serverStatus");
const emptyState = document.querySelector("#emptyState");
const loadingState = document.querySelector("#loadingState");
const resultView = document.querySelector("#resultView");
const lastSubmittedClaim = document.querySelector("#lastSubmittedClaim");

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.mode = button.dataset.mode;
  });
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    claimInput.value = button.dataset.sample;
    claimInput.dispatchEvent(new Event("input", { bubbles: true }));
    claimInput.focus();
  });
});

claimInput.addEventListener("input", handleClaimInput);
claimInput.addEventListener("keydown", handleClaimKeydown);
verifyBtn.addEventListener("click", verifyClaim);
clearBtn.addEventListener("click", resetWorkspace);

boot();
startStarfield();

async function boot() {
  try {
    const data = await apiGet("/api/health");
    serverStatus.textContent = data.message || "Backend ready";
    serverStatus.classList.add("ready");
  } catch (error) {
    serverStatus.textContent = "Backend offline";
    serverStatus.classList.add("error");
  }
}

async function verifyClaim() {
  const claim = normalizeClaimText(claimInput.value);
  if (!claim) {
    claimInput.focus();
    return;
  }
  clearTimeout(state.autoVerifyTimer);
  const verificationToken = state.verificationToken + 1;
  state.verificationToken = verificationToken;
  setLoading(true);
  try {
    const data = await apiPost("/api/verify", { claim, mode: state.mode });
    if (verificationToken !== state.verificationToken) {
      return;
    }
    state.latestResult = data.result;
    state.lastSubmittedClaim = claim;
    renderLastSubmittedClaim(claim);
    renderDashboard(data.result);
  } catch (error) {
    if (verificationToken !== state.verificationToken) {
      return;
    }
    renderError(error);
  } finally {
    if (verificationToken === state.verificationToken) {
      setLoading(false);
    }
  }
}

function handleClaimInput() {
  const claim = claimInput.value;
  if (shouldClearStaleResult(claim, state.lastSubmittedClaim, Boolean(state.latestResult))) {
    clearCurrentResult();
  }
  clearTimeout(state.autoVerifyTimer);
  if (!shouldAutoVerifyOnInput(claim, state.lastSubmittedClaim)) {
    return;
  }
  state.autoVerifyTimer = setTimeout(() => {
    if (shouldAutoVerifyOnInput(claimInput.value, state.lastSubmittedClaim)) {
      verifyClaim();
    }
  }, 700);
}

function handleClaimKeydown(event) {
  if (!shouldSubmitOnEnter(event)) {
    return;
  }
  event.preventDefault();
  verifyClaim();
}

function renderDashboard(result) {
  showResultShell();
  const metrics = deriveMetrics(result);
  renderVerdict(result, metrics);
  renderAnalytics(result, metrics);
  renderExplanation(result);
  renderDecomposition(result);
  renderReliabilityChart(result);
  renderStanceChart(result, metrics);
  renderConfidenceBreakdown(result);
  renderEvidenceColumns(result);
  renderSourceCards(result);
  renderDiagnostics(result);
  renderSideSummary(metrics);
}

function renderLastSubmittedClaim(claim) {
  if (!lastSubmittedClaim) {
    return;
  }
  const label = formatLastSubmittedClaim(claim);
  lastSubmittedClaim.textContent = label;
  lastSubmittedClaim.classList.toggle("hidden", !label);
}

function renderVerdict(result, metrics) {
  const confidence = result.confidence || {};
  const percentage = confidence.percentage;
  document.querySelector("#verdictText").textContent = result.verdict || "Insufficient Evidence";
  document.querySelector("#statusText").textContent = `${result.status || "UNKNOWN"} • ${result.claim_id || "No ID yet"}`;
  document.querySelector("#confidenceValue").textContent = percentage === null || percentage === undefined ? "N/A" : `${percentage}%`;
  document.querySelector("#confidenceGauge").style.setProperty("--confidence", percentage || 0);
  document.querySelector("#agreementValue").textContent = `${metrics.agreement}%`;
  document.querySelector("#verdictTags").innerHTML = [
    tag(result.processed_query?.category || "General"),
    tag(result.processed_query?.intent || "fact check"),
    tag(`quality ${result.quality_metrics?.overall_verification_quality || 0}%`),
    tag(`${metrics.averageCredibility}/10 avg credibility`),
  ].join("");
}

function renderAnalytics(result, metrics) {
  document.querySelector("#sourcesAnalyzed").textContent = metrics.sourcesAnalyzed;
  document.querySelector("#supportingSources").textContent = result.supporting_sources || 0;
  document.querySelector("#contradictingSources").textContent = result.contradicting_sources || 0;
  document.querySelector("#coverageScore").textContent = `${result.quality_metrics?.coverage_score || 0}%`;
  document.querySelector("#independentSources").textContent = result.retrieval_analysis?.independent_source_count || 0;
  document.querySelector("#duplicateClusters").textContent = result.retrieval_analysis?.duplicate_source_clusters || 0;
}

function renderExplanation(result) {
  const explanation = result.explanation || "No explanation generated.";
  document.querySelector("#explanationText").innerHTML = escapeText(explanation).replaceAll("\n", "<br>");
}

function renderDecomposition(result) {
  const processed = result.processed_query || {};
  const entities = processed.entities || {};
  const representation = processed.claim_representation || {};
  const frame = processed.pipeline_summary?.claim_structure || processed.nlp_pipeline?.claim_structure || {};
  const eventTerms = representation.events || inferEvents(processed.tokens || []);
  const stages = processed.pipeline_summary?.stages || processed.nlp_pipeline?.stages || [];
  document.querySelector("#claimDecomposition").innerHTML = `
    <div class="entity-grid">
      ${entityRow("Claim type", [representation.claim_type || processed.category || "General"])}
      ${entityRow("People", entities.people)}
      ${entityRow("Organizations", entities.organizations)}
      ${entityRow("Locations", representation.locations || [...(entities.cities || []), ...(entities.countries || [])])}
      ${entityRow("Dates", representation.dates || entities.dates)}
      ${entityRow("Events", eventTerms)}
      ${entityRow("Numbers", representation.numbers || entities.numbers)}
      ${entityRow("Units", representation.units || [])}
      ${entityRow("Subject", [representation.subject || frame.subject || "Not detected"])}
      ${entityRow("Predicate", [representation.relation || frame.predicate || "Not detected"])}
      ${entityRow("Object", [representation.object || frame.object || "Not detected"])}
    </div>
  `;
  document.querySelector("#nlpStages").innerHTML = stages.length
    ? stages.map((stage) => tag(stage.replaceAll("_", " "))).join("")
    : emptyText("No NLP stages available.");
}

function renderReliabilityChart(result) {
  const sources = result.sources || [];
  const buckets = [
    { label: "High", count: sources.filter((item) => (item.reliability || 0) >= 8).length, color: "var(--mint)" },
    { label: "Medium", count: sources.filter((item) => (item.reliability || 0) >= 5 && (item.reliability || 0) < 8).length, color: "var(--amber)" },
    { label: "Low", count: sources.filter((item) => (item.reliability || 0) < 5).length, color: "var(--red)" },
  ];
  const max = Math.max(1, ...buckets.map((item) => item.count));
  document.querySelector("#reliabilityChart").innerHTML = buckets.map((item) => `
    <div class="bar-row">
      <span class="source-meta">${item.label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.round((item.count / max) * 100)}%; background:${item.color}"></div></div>
      <strong>${item.count}</strong>
    </div>
  `).join("") || emptyText("No sources available.");
}

function renderStanceChart(result, metrics) {
  const support = result.supporting_sources || 0;
  const contradict = result.contradicting_sources || 0;
  const neutral = result.neutral_sources || 0;
  const total = Math.max(1, support + contradict + neutral);
  document.querySelector("#stanceChart").innerHTML = `
    <div class="stance-strip">
      <span class="stance-support" style="width:${(support / total) * 100}%"></span>
      <span class="stance-contradict" style="width:${(contradict / total) * 100}%"></span>
      <span class="stance-neutral" style="width:${(neutral / total) * 100}%"></span>
    </div>
    <div class="tag-row">
      ${tag(`support ${support}`)}
      ${tag(`contradict ${contradict}`)}
      ${tag(`neutral ${neutral}`)}
      ${tag(`agreement ${metrics.agreement}%`)}
    </div>
  `;
}

function renderConfidenceBreakdown(result) {
  const breakdown = result.confidence?.breakdown || result.quality_metrics?.confidence_inputs || {};
  const rows = [
    { label: "Evidence strength", value: breakdown.evidence_strength || 0, color: "var(--mint)" },
    { label: "Source credibility", value: breakdown.source_credibility || 0, color: "var(--cyan)" },
    { label: "Agreement", value: breakdown.cross_source_agreement || breakdown.agreement || 0, color: "var(--amber)" },
    { label: "Claim coverage", value: breakdown.claim_coverage || breakdown.coverage || 0, color: "var(--pink)" },
  ];
  document.querySelector("#confidenceBreakdown").innerHTML = rows.map((item) => `
    <div class="bar-row">
      <span class="source-meta">${item.label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, item.value))}%; background:${item.color}"></div></div>
      <strong>${item.value}%</strong>
    </div>
  `).join("");
}

function renderEvidenceColumns(result) {
  const evidence = result.evidence || [];
  const supporting = evidence.filter((item) => item.stance === "support");
  const contradicting = evidence.filter((item) => item.stance === "contradict");
  document.querySelector("#supportingEvidence").innerHTML = supporting.length
    ? supporting.slice(0, 4).map(renderEvidenceCard).join("")
    : emptyText("No directly supporting evidence was extracted.");
  document.querySelector("#contradictingEvidence").innerHTML = contradicting.length
    ? contradicting.slice(0, 4).map(renderEvidenceCard).join("")
    : emptyText("No directly contradicting evidence was extracted.");
}

function renderEvidenceCard(item) {
  return `
    <article class="evidence-card">
      <p>${escapeText(item.sentence || "")}</p>
      <div class="tag-row">
        ${tag(item.domain || "unknown source")}
        ${tag(`reliability ${item.reliability || 0}/10`)}
        ${tag(`coverage ${item.claim_coverage || 0}%`)}
        ${tag(`jaccard ${item.jaccard_similarity || 0}%`)}
        ${tag(`cosine ${item.cosine_similarity || 0}%`)}
      </div>
    </article>
  `;
}

function renderSourceCards(result) {
  const sources = result.sources || [];
  document.querySelector("#sourceCards").innerHTML = sources.length
    ? sources.slice(0, 8).map((source, index) => `
      <article class="source-card">
        <header>
          <div>
            <h4>${escapeText(source.title || source.domain || `Source ${index + 1}`)}</h4>
            <span class="source-meta">${escapeText(source.domain || "unknown domain")}</span>
          </div>
          <span class="stance-pill ${escapeText(source.stance || "neutral")}">${escapeText(source.stance || "neutral")}</span>
        </header>
        <p>${escapeText(source.evidence || source.rejection_reason || "No extracted snippet available.")}</p>
        <div class="tag-row">
          ${tag(`Reliability ${source.reliability || 0}/10`)}
          ${tag(source.credibility_tier || "Tier n/a")}
          ${tag(`Quality ${source.extraction_quality || 0}%`)}
          ${tag(source.accepted ? "Accepted" : "Rejected")}
        </div>
        ${source.url ? `<p><a class="source-link" href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">Open citation</a></p>` : ""}
      </article>
    `).join("")
    : emptyText("No source cards available.");
  renderSourceList(sources);
}

function renderSourceList(sources) {
  const target = document.querySelector("#sourceList");
  if (!target) return;
  target.innerHTML = sources.length
    ? sources.map((source, index) => `
      <div class="source-list-row">
        <span class="source-index">${index + 1}</span>
        <div>
          <strong>${escapeText(source.title || source.domain || `Source ${index + 1}`)}</strong>
          <p class="source-meta">${escapeText(source.domain || "unknown domain")} • ${escapeText(source.stance || "neutral")} • reliability ${source.reliability || 0}/10 • ${source.accepted ? "accepted" : "rejected"}</p>
          ${source.url ? `<a class="source-link" href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">${escapeText(source.url)}</a>` : ""}
        </div>
      </div>
    `).join("")
    : emptyText("No sources analyzed yet.");
}

function renderDiagnostics(result) {
  const retrieval = result.retrieval_analysis || {};
  const diagnostics = retrieval.search_diagnostics || {};
  const attempts = diagnostics.fallback_attempts || [];
  document.querySelector("#diagnosticsPanel").innerHTML = `
    <div class="diagnostic-grid">
      ${diagnostic("Primary status", diagnostics.status_code || "N/A")}
      ${diagnostic("RSS/secondary status", diagnostics.secondary_status_code || "N/A")}
      ${diagnostic("Search results", diagnostics.results_returned || 0)}
      ${diagnostic("Pages retrieved", retrieval.pages_retrieved || 0)}
      ${diagnostic("Pages accepted", retrieval.sources_successfully_retrieved || 0)}
      ${diagnostic("Extraction success", `${retrieval.extraction_success_rate || 0}%`)}
    </div>
    <article class="diagnostic-card">
      <p><strong>Primary search URL</strong></p>
      <p class="source-meta">${escapeText(diagnostics.search_url || "None")}</p>
      ${diagnostics.secondary_search_url ? `<p><strong>Secondary search URL</strong></p><p class="source-meta">${escapeText(diagnostics.secondary_search_url)}</p>` : ""}
      ${attempts.length ? `<p><strong>Fallback attempts</strong></p>${attempts.map((item) => `<p class="source-meta">${escapeText(item.query)} • status ${escapeText(String(item.status_code || "N/A"))} • results ${escapeText(String(item.results_returned || 0))}</p>`).join("")}` : ""}
    </article>
  `;
}

function renderSideSummary(metrics) {
  document.querySelector("#sideSources").textContent = metrics.sourcesAnalyzed;
  document.querySelector("#sideCoverage").textContent = `${metrics.coverage}%`;
  document.querySelector("#sideCredibility").textContent = `${metrics.averageCredibility}/10`;
}

function renderError(error) {
  showResultShell();
  document.querySelector("#verdictText").textContent = "ERROR";
  document.querySelector("#statusText").textContent = "Request failed";
  document.querySelector("#confidenceValue").textContent = "N/A";
  document.querySelector("#explanationText").textContent = error.message || String(error);
}

function clearCurrentResult() {
  state.latestResult = null;
  resultView.classList.add("hidden");
  loadingState.classList.add("hidden");
  emptyState.classList.remove("hidden");
}

function deriveMetrics(result) {
  const sources = result.sources || [];
  const support = result.supporting_sources || 0;
  const contradict = result.contradicting_sources || 0;
  const neutral = result.neutral_sources || 0;
  const stanceTotal = support + contradict + neutral;
  const strongest = Math.max(support, contradict);
  const averageCredibility = sources.length
    ? Math.round((sources.reduce((sum, item) => sum + (item.reliability || 0), 0) / sources.length) * 10) / 10
    : 0;
  return {
    sourcesAnalyzed: sources.length,
    averageCredibility,
    agreement: stanceTotal ? Math.round((strongest / stanceTotal) * 100) : 0,
    coverage: result.quality_metrics?.coverage_score || 0,
  };
}

function inferEvents(tokens) {
  const eventHints = new Set(["attack", "war", "growth", "forecast", "pandemic", "election", "launch", "mission", "conflict", "disruption", "discover", "reduced"]);
  return (tokens || []).filter((token) => eventHints.has(token)).slice(0, 8);
}

function entityRow(label, values) {
  const cleaned = (values || []).filter(Boolean);
  return `
    <div class="entity-row">
      <span class="entity-label">${escapeText(label)}</span>
      <strong class="entity-values">${escapeText(cleaned.length ? cleaned.join(", ") : "None detected")}</strong>
    </div>
  `;
}

function diagnostic(label, value) {
  return `<div class="diagnostic-card"><span class="source-meta">${escapeText(label)}</span><strong>${escapeText(String(value))}</strong></div>`;
}

function tag(value) {
  return `<span class="tag">${escapeText(value)}</span>`;
}

function emptyText(value) {
  return `<article class="evidence-card"><p>${escapeText(value)}</p></article>`;
}

function resetWorkspace() {
  clearTimeout(state.autoVerifyTimer);
  claimInput.value = "";
  state.latestResult = null;
  state.lastSubmittedClaim = "";
  state.verificationToken += 1;
  renderLastSubmittedClaim("");
  resultView.classList.add("hidden");
  loadingState.classList.add("hidden");
  emptyState.classList.remove("hidden");
  renderSideSummary({ sourcesAnalyzed: 0, coverage: 0, averageCredibility: 0 });
}

function setLoading(active) {
  verifyBtn.disabled = active;
  verifyBtn.lastChild.textContent = active ? " Verifying" : " Verify Claim";
  if (active) {
    emptyState.classList.add("hidden");
    resultView.classList.add("hidden");
    loadingState.classList.remove("hidden");
  } else {
    loadingState.classList.add("hidden");
  }
}

function showResultShell() {
  emptyState.classList.add("hidden");
  loadingState.classList.add("hidden");
  resultView.classList.remove("hidden");
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  return parseResponse(response);
}

async function apiPost(path, payload) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

async function parseResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function escapeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeText(value).replaceAll("`", "&#096;");
}

function startStarfield() {
  const canvas = document.querySelector("#starfield");
  const context = canvas.getContext("2d");
  const stars = [];
  let width = 0;
  let height = 0;
  let pixelRatio = 1;

  function resize() {
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * pixelRatio);
    canvas.height = Math.floor(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    stars.length = 0;
    const count = Math.floor((width * height) / 5600);
    for (let index = 0; index < count; index += 1) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 1.7 + 0.2,
        speed: Math.random() * 0.22 + 0.05,
        color: ["#ffffff", "#62e6ff", "#ff6aa8", "#ffd36e", "#52f3a9"][Math.floor(Math.random() * 5)],
      });
    }
  }

  function draw() {
    context.clearRect(0, 0, width, height);
    context.globalCompositeOperation = "lighter";
    stars.forEach((star) => {
      star.y += star.speed;
      star.x += Math.sin(star.y * 0.009) * 0.05;
      if (star.y > height + 8) {
        star.y = -8;
        star.x = Math.random() * width;
      }
      context.beginPath();
      context.globalAlpha = 0.32 + Math.sin(Date.now() * 0.002 + star.x) * 0.22;
      context.fillStyle = star.color;
      context.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
      context.fill();
    });
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  draw();
}
