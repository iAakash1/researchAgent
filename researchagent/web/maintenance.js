(function (root) {
  const changeTypes = ["kept","weakened","invalidated","needs_review","newly_contradicted","new"];
  const labels = {kept:"Kept",weakened:"Weakened",invalidated:"Invalidated",needs_review:"Needs review",newly_contradicted:"Newly contradicted",new:"New"};
  const doiPattern = /^10\.\d{4,9}\/\S+$/i;

  function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]); }
  function doiLink(doi) {
    const value = String(doi ?? "").trim();
    if (!doiPattern.test(value)) return value ? escapeHtml(value) : "";
    return `<a href="https://doi.org/${escapeHtml(encodeURI(value))}" target="_blank" rel="noopener noreferrer">${escapeHtml(value)}</a>`;
  }
  function trigger(item) {
    const parts = [escapeHtml(item.kind)];
    if (item.status) parts.push(escapeHtml(item.status));
    if (item.paper_id) parts.push(escapeHtml(item.paper_id));
    if (item.doi) parts.push(`DOI ${doiLink(item.doi)}`);
    if (item.notice_doi) parts.push(`notice ${doiLink(item.notice_doi)}`);
    if (item.notice_date) parts.push(escapeHtml(item.notice_date));
    if (item.registry) parts.push(escapeHtml(item.registry));
    return `<li>${parts.join(" · ")}<br><span class="muted">${escapeHtml(item.detail)}</span></li>`;
  }
  function usage(diff) {
    const u = diff.usage || {};
    const cost = u.estimated_cost_usd == null ? "Unavailable" : `$${Number(u.estimated_cost_usd).toFixed(6)}`;
    return `<p class="meta">${escapeHtml(diff.mode)} · ${diff.affected_conclusions} of ${diff.total_conclusions} conclusions affected · ${diff.papers_checked} papers checked · ${u.total_tokens ?? 0} tokens · est. cost ${cost} · ${Math.round(u.latency_ms ?? 0)} ms</p>`;
  }
  function renderDiffPanel(diff) {
    if (!diff) return `<h2>Conclusion maintenance</h2><p class="muted">Not maintained yet. Run a check to look for retractions, corrections and new papers.</p>`;
    const counts = diff.counts || {};
    const metrics = changeTypes.map(type => `<div class="metric" data-change="${type}"><strong>${counts[type] ?? 0}</strong>${labels[type]}</div>`).join("");
    const changes = (diff.changes || []).map(change => `<article class="card"><span class="badge ${change.change_type === "kept" || change.change_type === "new" ? "" : "warn"}">${escapeHtml(labels[change.change_type] || change.change_type)}</span><h3>${escapeHtml(change.statement || change.conclusion_id)}</h3><p class="meta">${escapeHtml(change.conclusion_id)} · v${change.before_version ?? "–"} → v${change.after_version} · −${(change.removed_evidence_ids || []).length} / +${(change.added_evidence_ids || []).length} evidence</p><ul class="triggers">${(change.triggers || []).map(trigger).join("")}</ul></article>`).join("");
    const body = changes ? `<div class="grid">${changes}</div>` : `<p class="muted">No conclusion changed: no rule fired.</p>`;
    const failures = (diff.lookup_failures || []).length ? `<p class="meta">${diff.lookup_failures.length} integrity lookup(s) failed and were treated as unknown, never as current.</p>` : "";
    return `<h2>Conclusion maintenance</h2>${usage(diff)}<div class="metrics">${metrics}</div>${failures}${body}`;
  }

  const api = {changeTypes, doiLink, renderDiffPanel};
  root.ResearchMaintenance = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
