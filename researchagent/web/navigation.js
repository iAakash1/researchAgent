(function (root) {
  // The start form and the run page are two sections of the same /app/ page, switched with
  // the `hidden` class. The active run is remembered in localStorage and reattached on load,
  // so going "home" must forget it here or the run page would simply come back.
  const storageKey = "researchagent.activeRun";
  const panels = ["#overview","#papers","#evidence","#findings","#contradictions","#gaps","#maintenance"];

  // Back to an empty start form. Closes this page's progress stream and forgets the run in
  // this browser; it sends nothing to the server, so a running run keeps running there.
  function resetToStart(doc, {storage=null, source=null, key=storageKey}={}) {
    if (source && typeof source.close === "function") source.close();
    try { if (storage) storage.removeItem(key); } catch { /* storage unavailable */ }
    const el = selector => doc.querySelector(selector);
    const form = el("#research-form");
    if (form) {
      form.reset();
      const submit = form.querySelector("button");
      if (submit) submit.disabled = false;
    }
    el("#run")?.classList.add("hidden");
    el("#results")?.classList.add("hidden");
    el("#new-research")?.classList.remove("hidden");
    const error = el("#error");
    if (error) { error.textContent = ""; error.classList.add("hidden"); }
    for (const id of panels) { const panel = el(id); if (panel) panel.innerHTML = ""; }
    const tabs = el(".tabs"); if (tabs) tabs.innerHTML = "";
    const title = el("#run-title"); if (title) title.textContent = "";
    const status = el("#run-status"); if (status) status.textContent = "Running";
    for (const step of el("#progress")?.children ?? []) step.className = "";
    el("#goal")?.focus();
    return true;
  }

  // Plain left clicks are handled in place; modified clicks keep the browser's own behaviour.
  function isPlainClick(event) {
    return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey;
  }

  const api = {storageKey, resetToStart, isPlainClick};
  root.ResearchNavigation = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
