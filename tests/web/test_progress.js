const test=require("node:test");
const assert=require("node:assert/strict");
const {phases,classesAt}=require("../../researchagent/web/progress.js");

function expectFailure(stage,expectedIndex){
  const classes=classesAt(stage,true);
  assert.equal(classes[expectedIndex],"failed");
  assert.deepEqual(classes.slice(expectedIndex+1),Array(phases.length-expectedIndex-1).fill(""));
  assert.notEqual(classes.at(-1),"active");
  assert.notEqual(classes.at(-1),"done");
}

test("failure at Planning stops at Planning",()=>expectFailure("planning",0));
test("failure at Reasoning stops at Reasoning",()=>expectFailure("reasoning",5));
test("failure at Verification stops at Verification",()=>expectFailure("verification",6));
test("success reaches Complete",()=>{
  const classes=classesAt("Complete",false);
  assert.equal(classes.at(-1),"active");
  assert.ok(classes.slice(0,-1).every(value=>value==="done"));
});
test("unknown events do not change progress",()=>{
  assert.deepEqual(classesAt("unknown",false),Array(phases.length).fill(""));
});
