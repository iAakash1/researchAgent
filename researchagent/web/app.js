const {phases,phaseForStage,classesAt}=ResearchProgress;
const phaseByEvent = {"discovery.completed":"Searching papers","paper.discovered":"Searching papers","paper.acquired":"Acquiring papers","document.loaded":"Processing papers","document.parsed":"Processing papers","knowledge.extracted":"Extracting evidence","evidence.indexed":"Extracting evidence","bundle.created":"Extracting evidence","research.iteration.started":"Reasoning","research.finding.created":"Reasoning","research.finding.verified":"Verification","research.finding.rejected":"Verification","research.terminated":"Review"};
const phaseByAgent = {"planner":"Planning","retrieval":"Reasoning","reasoning":"Reasoning","verification":"Verification","reviewer":"Review"};
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]);
const progress = document.querySelector("#progress");
let highestStageIndex=0;
phases.forEach(name => progress.insertAdjacentHTML("beforeend", `<li data-phase="${name}">${name}</li>`));
function paintPhase(index,failed=false){ const classes=classesAt(phases[index],failed); [...progress.children].forEach((el,i)=>{el.className=classes[i];}); }
function setPhase(name){ const index=phases.indexOf(name); if(index<highestStageIndex||index<0)return; highestStageIndex=index; paintPhase(index); }
function setFailedPhase(name){ const index=phases.indexOf(name); if(index<0)return; highestStageIndex=index; paintPhase(index,true); }
function cards(items, render){ return items.length ? `<div class="grid">${items.map(render).join("")}</div>` : `<p class="muted">None recorded for this run.</p>`; }
function externalUrl(value){ try{const url=new URL(value);return ["http:","https:"].includes(url.protocol)?url.href:null;}catch{return null;} }
function paperTitle(paper){ const url=externalUrl(paper.source_url); return url?`<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(paper.title)}</a>`:escapeHtml(paper.title); }
function paperPdf(paper){ const url=externalUrl(paper.pdf_url); return url?`<a class="paper-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">PDF</a>`:""; }
function modelSummary(usage){ if(!usage?.calls?.length)return ""; const provider=usage.providers.join(", "),model=usage.models.join(", "),cost=usage.estimated_cost_usd==null?"Unavailable":`$${usage.estimated_cost_usd.toFixed(6)}`; return `<div class="model-summary"><span>Provider <strong>${escapeHtml(provider)}</strong></span><span>Model <strong>${escapeHtml(model)}</strong></span><span>Fallback <strong>${usage.fallback_used?"Yes":"No"}</strong></span><span>Tokens <strong>${usage.total_tokens}</strong></span><span>Est. API cost <strong>${cost}</strong></span></div>`; }
function render(result,run=null){
  const failed=result.status==="failed"||Boolean(result.failure),failedPhase=phaseForStage(result.progress_stage)||phases[Math.min(run?.stageIndex??highestStageIndex,phases.length-2)];
  document.querySelector("#run-status").textContent=failed?"Failed":"Complete";
  if(failed){setFailedPhase(failedPhase);}
  else{setPhase("Complete");}
  if(run){run.status=failed?"failed":"completed";run.stageIndex=highestStageIndex;run.failedStage=failed?failedPhase:null;saveRun(run);}
  const metrics=[[result.discovered_papers,"Discovered"],[result.selected_papers,"Selected"],[result.processed_papers,"Processed"],[result.findings.filter(f=>f.reviewer_accepted).length,"Accepted"]];
  document.querySelector("#overview").innerHTML=`<h2>Overview</h2><div class="metrics">${metrics.map(([v,l])=>`<div class="metric"><strong>${v}</strong>${l}</div>`).join("")}</div>${modelSummary(result.model_usage)}<h3>Final synthesis</h3><p>${escapeHtml(result.final_summary)}</p>`;
  document.querySelector("#papers").innerHTML=`<h2>Papers</h2>${cards(result.papers,p=>`<article class="card"><h3>${paperTitle(p)}</h3><p class="meta">${escapeHtml(p.provider)} · ${p.year||"Year unknown"}${p.doi?` · DOI ${escapeHtml(p.doi)}`:""} · relevance ${p.relevance_score.toFixed(3)} · ${escapeHtml(p.relevance_decision)}</p><span class="badge ${p.processed?'':'warn'}">${p.processed?'Processed':p.selected?(p.accessible?'Accessible':'Unavailable'):'Discovered'}</span>${paperPdf(p)}${p.failure_reason?`<p>${escapeHtml(p.failure_reason)}</p>`:""}</article>`)}`;
  document.querySelector("#evidence").innerHTML=`<h2>Evidence</h2>${cards(result.evidence,e=>`<article class="card"><p class="quote">“${escapeHtml(e.quote)}”</p><p class="citation">${escapeHtml(e.paper_id)} · ${escapeHtml(e.location)}</p></article>`)}`;
  document.querySelector("#findings").innerHTML=`<h2>Findings</h2>${cards(result.findings,f=>`<article class="card"><span class="badge ${f.reviewer_accepted?'':'warn'}">${escapeHtml(f.verification_status||f.status)}</span><h3>${escapeHtml(f.statement)}</h3><p class="meta">Supporting papers: ${escapeHtml(f.supporting_sources.join(", ")||"none")}</p>${f.provenance.map(p=>`<p class="citation">${escapeHtml(p)}</p>`).join("")}</article>`)}`;
  document.querySelector("#contradictions").innerHTML=`<h2>Contradictions & disagreements</h2>${cards(result.contradictions,c=>`<article class="card"><h3>${escapeHtml(c.description)}</h3><p><strong>${escapeHtml(c.left_paper_id)}</strong>: ${escapeHtml(c.left_quotes.join(" "))}</p><p><strong>${escapeHtml(c.right_paper_id)}</strong>: ${escapeHtml(c.right_quotes.join(" "))}</p></article>`)}`;
  document.querySelector("#gaps").innerHTML=`<h2>Research gaps & limitations</h2>${[...result.research_gaps,...result.limitations].length?`<ul>${[...result.research_gaps,...result.limitations].map(x=>`<li>${escapeHtml(x)}</li>`).join("")}</ul>`:`<p class="muted">No explicit gaps recorded.</p>`}`;
  const sections=["Overview","Papers","Evidence","Findings","Contradictions","Gaps"];
  document.querySelector(".tabs").innerHTML=sections.map(x=>`<a href="#${x.toLowerCase()}">${x}</a>`).join("");
  document.querySelector("#results").classList.remove("hidden");
}
const storageKey="researchagent.activeRun";
function showRun(goal,stageIndex=0,status="running",failedStage=null){ document.querySelector("#new-research").classList.add("hidden"); document.querySelector("#run").classList.remove("hidden"); document.querySelector("#run-title").textContent=goal; document.querySelector("#run-status").textContent=status==="failed"?"Failed":"Running"; highestStageIndex=Math.max(0,Math.min(stageIndex,phases.length-1)); if(status==="failed")setFailedPhase(failedStage||phases[Math.min(highestStageIndex,phases.length-2)]);else paintPhase(highestStageIndex); }
function saveRun(run){ localStorage.setItem(storageKey,JSON.stringify(run)); }
function loadRun(){ try{return JSON.parse(localStorage.getItem(storageKey));}catch{localStorage.removeItem(storageKey);return null;} }
function fail(message,stage=phases[highestStageIndex]){ document.querySelector("#run-status").textContent="Failed"; setFailedPhase(stage); const box=document.querySelector("#error"); box.textContent=message; box.classList.remove("hidden"); }
function handleProgress(item,run){ const phase=phaseByAgent[item.payload?.agent]||phaseByEvent[item.type]; if(!phase)return; const index=phases.indexOf(phase); if(item.type==="agent.failed"){run.status="failed";run.failedStage=phase;run.stageIndex=index;saveRun(run);setFailedPhase(phase);return;} if(index>(run.stageIndex||0)){run.stageIndex=index;saveRun(run);} setPhase(phase); }
function attach(run){
  const source=new EventSource(`/research/runs/${encodeURIComponent(run.runId)}/stream?after=${run.lastSequence||0}`);
  source.onopen=()=>{if(run.status!=="failed")document.querySelector("#run-status").textContent="Running";};
  source.addEventListener("progress",message=>{ const sequence=Number(message.lastEventId); if(sequence&&sequence<=(run.lastSequence||0))return; if(sequence)run.lastSequence=sequence; saveRun(run); handleProgress(JSON.parse(message.data),run); });
  source.addEventListener("result",message=>{ render(JSON.parse(message.data),run); source.close(); });
  source.addEventListener("failed",message=>{ run.status="failed"; run.failedStage=phases[run.stageIndex||0]; saveRun(run); fail(JSON.parse(message.data).message,run.failedStage); source.close(); });
  source.onerror=()=>{if(run.status!=="failed")document.querySelector("#run-status").textContent="Reconnecting";};
}
async function restore(){
  const run=loadRun(); if(!run?.runId||!run.goal)return; showRun(run.goal,run.stageIndex||0,run.status,run.failedStage);
  const response=await fetch(`/research/results/${encodeURIComponent(run.runId)}`);
  if(response.status===404){localStorage.removeItem(storageKey);window.location.reload();return;}
  if(!response.ok)throw new Error(`Could not restore run (${response.status})`);
  const snapshot=await response.json();
  if(snapshot.result)render(snapshot.result,run); else if(snapshot.error)fail(snapshot.error,run.failedStage||phases[run.stageIndex||0]); else attach(run);
}
document.querySelector("#research-form").addEventListener("submit",async event=>{
  event.preventDefault(); const button=event.target.querySelector("button"); button.disabled=true;
  const goal=document.querySelector("#goal").value.trim(); showRun(goal);
  const year=Number(document.querySelector("#year-from").value)||null;
  const payload={goal,constraints:{max_research_questions:Number(document.querySelector("#max-questions").value),year_from:year}};
  try{
    const response=await fetch("/research/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); if(!response.ok)throw new Error(`Request failed (${response.status})`);
    const created=await response.json(), run={runId:created.run_id,goal,lastSequence:0,stageIndex:0,status:created.status}; saveRun(run); attach(run);
  }catch(error){fail(error.message);}
  finally{button.disabled=false;}
});
restore().catch(error=>fail(error.message));
