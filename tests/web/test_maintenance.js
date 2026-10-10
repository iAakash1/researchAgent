const test=require("node:test");
const assert=require("node:assert/strict");
const {changeTypes,doiLink,renderDiffPanel}=require("../../researchagent/web/maintenance.js");

const diff={
  mode:"incremental",affected_conclusions:2,total_conclusions:3,papers_checked:3,
  counts:{kept:1,weakened:0,invalidated:1,needs_review:0,newly_contradicted:0,new:0},
  usage:{total_tokens:240,estimated_cost_usd:null,latency_ms:12.4},
  lookup_failures:[],
  changes:[{conclusion_id:"C-solo",statement:"Circuit breakers <b>shed</b> load.",before_version:1,after_version:2,change_type:"invalidated",removed_evidence_ids:["E1"],added_evidence_ids:[],
    triggers:[{kind:"integrity",status:"retracted",paper_id:"doi:10.5555/p1",doi:"10.5555/p1",notice_doi:"10.5555/notice-p1",notice_date:"2026-09-30",registry:"crossref",detail:"crossref lists retraction"}]}],
};

test("counts are shown for every change type",()=>{
  const html=renderDiffPanel(diff);
  for(const type of changeTypes)assert.match(html,new RegExp(`data-change="${type}"`));
  assert.match(html,/<strong>1<\/strong>Invalidated/);
});
test("the trigger DOI is a clickable doi.org link",()=>{
  const html=renderDiffPanel(diff);
  assert.match(html,/href="https:\/\/doi\.org\/10\.5555\/p1" target="_blank" rel="noopener noreferrer"/);
  assert.match(html,/notice <a href="https:\/\/doi\.org\/10\.5555\/notice-p1"/);
});
test("a value that is not a DOI is never linked",()=>{
  assert.equal(doiLink("javascript:alert(1)"),"javascript:alert(1)");
  assert.equal(doiLink(""),"");
});
test("statements are escaped",()=>{
  assert.doesNotMatch(renderDiffPanel(diff),/<b>shed<\/b>/);
});
test("an empty diff says no rule fired",()=>{
  assert.match(renderDiffPanel({...diff,changes:[],counts:{}}),/No conclusion changed: no rule fired/);
});
test("unknown cost is shown as unavailable, not zero",()=>{
  assert.match(renderDiffPanel(diff),/est\. cost Unavailable/);
});
test("no diff yet is explained",()=>{
  assert.match(renderDiffPanel(null),/Not maintained yet/);
});
