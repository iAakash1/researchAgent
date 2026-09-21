const phases = ["Planning","Searching papers","Acquiring papers","Processing papers","Extracting evidence","Reasoning","Verification","Review","Complete"];
const phaseByEvent = {"agent.started":"Planning","discovery.completed":"Searching papers","paper.discovered":"Searching papers","paper.acquired":"Acquiring papers","document.loaded":"Processing papers","document.parsed":"Processing papers","knowledge.extracted":"Extracting evidence","evidence.indexed":"Extracting evidence","bundle.created":"Extracting evidence","research.iteration.started":"Reasoning","research.finding.created":"Reasoning","research.finding.verified":"Verification","research.finding.rejected":"Verification","research.terminated":"Review"};
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]);
const progress = document.querySelector("#progress");
phases.forEach(name => progress.insertAdjacentHTML("beforeend", `<li data-phase="${name}">${name}</li>`));
function setPhase(name){ const index=phases.indexOf(name); [...progress.children].forEach((el,i)=>{el.className=i<index?"done":i===index?"active":"";}); }
function cards(items, render){ return items.length ? `<div class="grid">${items.map(render).join("")}</div>` : `<p class="muted">None recorded for this run.</p>`; }
function render(result){
  document.querySelector("#run-status").textContent=result.failure?"Failed":"Complete"; setPhase("Complete");
  const metrics=[[result.discovered_papers,"Discovered"],[result.selected_papers,"Selected"],[result.processed_papers,"Processed"],[result.findings.filter(f=>f.reviewer_accepted).length,"Accepted"]];
  document.querySelector("#overview").innerHTML=`<h2>Overview</h2><div class="metrics">${metrics.map(([v,l])=>`<div class="metric"><strong>${v}</strong>${l}</div>`).join("")}</div><h3>Final synthesis</h3><p>${escapeHtml(result.final_summary)}</p>`;
  document.querySelector("#papers").innerHTML=`<h2>Papers</h2>${cards(result.papers,p=>`<article class="card"><h3>${escapeHtml(p.title)}</h3><p class="meta">${escapeHtml(p.provider)} · ${p.year||"Year unknown"} · score ${p.score.toFixed(3)}</p><span class="badge ${p.processed?'':'warn'}">${p.processed?'Processed':p.selected?(p.accessible?'Accessible':'Unavailable'):'Discovered'}</span>${p.failure_reason?`<p>${escapeHtml(p.failure_reason)}</p>`:""}</article>`)}`;
  document.querySelector("#evidence").innerHTML=`<h2>Evidence</h2>${cards(result.evidence,e=>`<article class="card"><p class="quote">“${escapeHtml(e.quote)}”</p><p class="citation">${escapeHtml(e.paper_id)} · ${escapeHtml(e.location)}</p></article>`)}`;
  document.querySelector("#findings").innerHTML=`<h2>Findings</h2>${cards(result.findings,f=>`<article class="card"><span class="badge ${f.reviewer_accepted?'':'warn'}">${escapeHtml(f.verification_status||f.status)}</span><h3>${escapeHtml(f.statement)}</h3><p class="meta">Supporting papers: ${escapeHtml(f.supporting_sources.join(", ")||"none")}</p>${f.provenance.map(p=>`<p class="citation">${escapeHtml(p)}</p>`).join("")}</article>`)}`;
  document.querySelector("#contradictions").innerHTML=`<h2>Contradictions & disagreements</h2>${cards(result.contradictions,c=>`<article class="card"><h3>${escapeHtml(c.description)}</h3><p><strong>${escapeHtml(c.left_paper_id)}</strong>: ${escapeHtml(c.left_quotes.join(" "))}</p><p><strong>${escapeHtml(c.right_paper_id)}</strong>: ${escapeHtml(c.right_quotes.join(" "))}</p></article>`)}`;
  document.querySelector("#gaps").innerHTML=`<h2>Research gaps & limitations</h2>${[...result.research_gaps,...result.limitations].length?`<ul>${[...result.research_gaps,...result.limitations].map(x=>`<li>${escapeHtml(x)}</li>`).join("")}</ul>`:`<p class="muted">No explicit gaps recorded.</p>`}`;
  const sections=["Overview","Papers","Evidence","Findings","Contradictions","Gaps"];
  document.querySelector(".tabs").innerHTML=sections.map(x=>`<a href="#${x.toLowerCase()}">${x}</a>`).join("");
  document.querySelector("#results").classList.remove("hidden");
}
document.querySelector("#research-form").addEventListener("submit",async event=>{
  event.preventDefault(); const button=event.target.querySelector("button"); button.disabled=true;
  const goal=document.querySelector("#goal").value.trim(); document.querySelector("#new-research").classList.add("hidden"); document.querySelector("#run").classList.remove("hidden"); document.querySelector("#run-title").textContent=goal; setPhase("Planning");
  const year=Number(document.querySelector("#year-from").value)||null;
  const payload={goal,constraints:{max_research_questions:Number(document.querySelector("#max-questions").value),year_from:year}};
  try{
    const response=await fetch("/research/run/stream",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); if(!response.ok||!response.body) throw new Error(`Request failed (${response.status})`);
    const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer="";
    while(true){ const {value,done}=await reader.read(); buffer+=decoder.decode(value||new Uint8Array(),{stream:!done}); const blocks=buffer.split("\n\n"); buffer=blocks.pop()||""; for(const block of blocks){ const kind=(block.match(/^event: (.+)$/m)||[])[1], data=(block.match(/^data: (.+)$/m)||[])[1]; if(!data)continue; const item=JSON.parse(data); if(kind==="progress"){ const agent=item.payload?.agent; setPhase(agent&&["reasoning","verification","reviewer"].includes(agent)?({reasoning:"Reasoning",verification:"Verification",reviewer:"Review"})[agent]:(phaseByEvent[item.type]||"Planning")); } else if(kind==="result") render(item); else if(kind==="error") throw new Error(item.message); } if(done)break; }
  }catch(error){ document.querySelector("#run-status").textContent="Failed"; const box=document.querySelector("#error"); box.textContent=error.message; box.classList.remove("hidden"); }
  finally{button.disabled=false;}
});
