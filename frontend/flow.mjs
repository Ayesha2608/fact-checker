export function normalizeClaimText(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

export function shouldClearStaleResult(currentClaim, lastSubmittedClaim, hasResult) {
  if (!hasResult) {
    return false;
  }
  const current = normalizeClaimText(currentClaim);
  const previous = normalizeClaimText(lastSubmittedClaim);
  return Boolean(current) && current !== previous;
}

export function formatLastSubmittedClaim(claim) {
  const normalized = normalizeClaimText(claim);
  if (!normalized) {
    return "";
  }
  return "Last submitted claim: " + normalized;
}

export function shouldAutoVerifyOnInput(currentClaim, lastSubmittedClaim) {
  const current = normalizeClaimText(currentClaim);
  const previous = normalizeClaimText(lastSubmittedClaim);
  return Boolean(current) && current !== previous;
}

export function shouldSubmitOnEnter(event) {
  return Boolean(event && event.key === "Enter" && !event.shiftKey);
}
