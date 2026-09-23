(function (root) {
  const phases = ["Planning","Searching papers","Acquiring papers","Processing papers","Extracting evidence","Reasoning","Verification","Review","Complete"];
  const phaseByStage = {planning:"Planning",discovery:"Searching papers",acquisition:"Acquiring papers",document_intelligence:"Processing papers",knowledge_extraction:"Extracting evidence",evidence_intelligence:"Extracting evidence",retrieval:"Reasoning",reasoning:"Reasoning",verification:"Verification",review:"Review",terminated:"Review"};

  function phaseForStage(stage) { return phaseByStage[stage] || null; }
  function classesAt(stage, failed=false) {
    const name=phaseForStage(stage)||stage,index=phases.indexOf(name);
    if(index<0)return phases.map(()=>"");
    return phases.map((_,i)=>i<index?"done":i===index?(failed?"failed":"active"):"");
  }

  const api={phases,phaseForStage,classesAt};
  root.ResearchProgress=api;
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
})(typeof globalThis!=="undefined"?globalThis:this);
