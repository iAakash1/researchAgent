const test=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const {storageKey,resetToStart,isPlainClick}=require("../../researchagent/web/navigation.js");

const web=path.join(__dirname,"../../researchagent/web");
const html=fs.readFileSync(path.join(web,"index.html"),"utf8");
const app=fs.readFileSync(path.join(web,"app.js"),"utf8");

// A minimal stand-in for the page: only what navigation touches.
function element(){
  const classes=new Set();
  return {textContent:"",innerHTML:"",className:"",disabled:false,focused:false,children:[],
    classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c)},
    focus(){this.focused=true;}};
}
function page(state){
  const nodes={};
  for(const id of ["#new-research","#run","#results","#error","#run-title","#run-status","#goal",".tabs","#progress",
    "#overview","#papers","#evidence","#findings","#contradictions","#gaps","#maintenance"])nodes[id]=element();
  const submit=element();
  const form=Object.assign(element(),{reset(){nodes["#goal"].value="";this.wasReset=true;},querySelector:()=>submit});
  nodes["#research-form"]=form;
  nodes["#goal"].value="Why do multi-agent LLM systems fail?";
  nodes["#new-research"].classList.add("hidden");
  nodes["#run-title"].textContent="Why do multi-agent LLM systems fail?";
  nodes["#progress"].children=["done","done","active"].map(name=>Object.assign(element(),{className:name}));
  for(const id of ["#overview","#papers","#evidence","#findings","#contradictions","#gaps","#maintenance"])nodes[id].innerHTML="<h2>filled</h2>";
  nodes[".tabs"].innerHTML="<a>Overview</a>";
  submit.disabled=true;
  if(state==="complete"){nodes["#run-status"].textContent="Complete";nodes["#error"].classList.add("hidden");}
  if(state==="failed"){nodes["#run-status"].textContent="Failed";nodes["#error"].textContent="Planner failed";nodes["#progress"].children[2].className="failed";}
  if(state==="running"){nodes["#run-status"].textContent="Running";nodes["#results"].classList.add("hidden");nodes["#error"].classList.add("hidden");}
  const stored=new Map([[storageKey,JSON.stringify({runId:"r1",goal:"g",status:state})]]);
  const storage={removeItem:key=>stored.delete(key),getItem:key=>stored.get(key)??null};
  const source={closed:false,close(){this.closed=true;}};
  return {doc:{querySelector:selector=>nodes[selector]??null},nodes,submit,storage,source};
}
function assertAtStart({nodes,submit,storage,source}){
  assert.ok(nodes["#run"].classList.contains("hidden"),"run page hidden");
  assert.ok(!nodes["#new-research"].classList.contains("hidden"),"start form shown");
  assert.ok(nodes["#results"].classList.contains("hidden"));
  assert.ok(nodes["#research-form"].wasReset);
  assert.equal(nodes["#goal"].value,"","form is empty");
  assert.ok(nodes["#goal"].focused,"form is ready to type into");
  assert.equal(submit.disabled,false);
  assert.equal(storage.getItem(storageKey),null,"the run is forgotten, so a reload does not bring it back");
  assert.ok(source.closed,"this page stops listening to the old run");
  assert.equal(nodes["#error"].textContent,"");
  assert.ok(nodes["#error"].classList.contains("hidden"));
  assert.equal(nodes["#run-status"].textContent,"Running");
  assert.ok(nodes["#progress"].children.every(step=>step.className===""));
  for(const id of ["#overview","#papers","#evidence","#findings","#contradictions","#gaps","#maintenance"])assert.equal(nodes[id].innerHTML,"",id);
  assert.equal(nodes[".tabs"].innerHTML,"");
}

test("app and navigation agree on where the active run is remembered",()=>{
  assert.match(app,new RegExp(`const storageKey="${storageKey}"`));
  assert.match(app,/resetToStart\(document,\{storage:localStorage,source:activeSource,key:storageKey\}\)/);
});
test("brand text in the header links to the start page",()=>{
  assert.match(html,/<header><a id="home-link" href="\/app\/">ResearchAgent<\/a>/);
  assert.match(app,/#home-link"\)\.addEventListener\("click"/);
});
test("the run page has a visible New research button",()=>{
  const run=html.slice(html.indexOf('<section id="run"'),html.indexOf("</main>"));
  assert.match(run,/<button type="button" id="new-research-button" class="back-button">← New research<\/button>/);
  assert.doesNotMatch(run.slice(0,run.indexOf("new-research-button")),/id="results"/,"button sits above the results, outside them");
  assert.match(app,/#new-research-button"\)\.addEventListener\("click",goHome\)/);
  assert.ok(html.indexOf("/app/navigation.js")<html.indexOf("/app/app.js"),"navigation loads before app");
});
test("navigation from a complete run returns to an empty start form",()=>{
  const p=page("complete"); resetToStart(p.doc,p); assertAtStart(p);
});
test("navigation from a failed run clears the failure and returns to the start form",()=>{
  const p=page("failed"); resetToStart(p.doc,p); assertAtStart(p);
});
test("navigation from a running run only stops listening; nothing is sent to cancel it",()=>{
  const p=page("running"); resetToStart(p.doc,p); assertAtStart(p);
  assert.doesNotMatch(fs.readFileSync(path.join(web,"navigation.js"),"utf8"),/fetch\(/);
});
test("unavailable storage does not block navigation",()=>{
  const p=page("complete"); p.storage.removeItem=()=>{throw new Error("blocked");};
  assert.equal(resetToStart(p.doc,{storage:p.storage,source:p.source}),true);
  assert.ok(!p.nodes["#new-research"].classList.contains("hidden"));
});
test("only plain left clicks are taken over; modified clicks keep browser behaviour",()=>{
  assert.equal(isPlainClick({button:0}),true);
  assert.equal(isPlainClick({button:0,metaKey:true}),false);
  assert.equal(isPlainClick({button:0,ctrlKey:true}),false);
  assert.equal(isPlainClick({button:1}),false);
});
