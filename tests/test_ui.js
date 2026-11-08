// Behavior tests against the browser script with a tiny DOM/transport harness.
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
function harness(){
 const elements=new Map();
 const doc={documentElement:{},querySelector(s){if(!elements.has(s))elements.set(s,{querySelectorAll:()=>[],appendChild(){},selectedOptions:[],options:[],children:[],classList:{toggle(){}},value:'',textContent:''});return elements.get(s);},querySelectorAll:()=>[],addEventListener(){}};
 const context=vm.createContext({document:doc,console,structuredClone,crypto:{randomUUID:()=>String(Math.random())},setInterval(){},setTimeout(){},URL,Blob,fetch:()=>new Promise(()=>{})});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../facade_next/static/app.js'),'utf8'),context);
 return {context,doc};
}
(async()=>{
 if(!process.argv.includes('--annotation')){
 const a=harness();
 vm.runInContext("state={csrf:'test'}; refresh=async()=>{throw Error('state refresh unavailable')};",a.context);
 let calls=0;
 a.context.fetch=async()=>{calls++;return {ok:true,json:async()=>({id:'created-once'})};};
 const one=await vm.runInContext("act('create_project',{name:'same'})",a.context).catch(e=>({error:e.message}));
 const two=await vm.runInContext("act('create_project',{name:'same'})",a.context).catch(e=>({error:e.message}));
 assert.equal(calls,1,'A refresh failure after accepted POST must not resubmit with a new ID');
 assert.equal(one.id,'created-once');assert.equal(two.id,'created-once');
 }
 const b=harness();b.context.steps=[];
 vm.runInContext("draft={rows:[{}]}; render=()=>{}; collectAnnotations=()=>steps.push('collect'); renderPhotos=()=>steps.push('redraw');",b.context);
 b.doc.querySelector('#language').onchange({target:{value:'en'}});
 assert.deepEqual(b.context.steps,['collect','redraw'],'Collect annotation edits before language redraw');
 const c=harness();
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../facade_next/static/chat.js'),'utf8'),c.context);
 vm.runInContext("state={projects:[{id:'P',name:'project',active:null,train_pool:[],eval_pool:[]}],versions:[],datasets:[],conversations:[{id:'C',project:'P',messages:[],card:{kind:'annotation',draft:{id:'D',rows:[{hollow:null,spalling:0,crack:0,path:'a.jpg'}]}}}]};projectId='P';collectAnnotations=()=>{};renderPhotos=()=>{};renderChat();draft.rows[0].hollow=1;lang='en';renderChat();",c.context);
 if(!process.argv.includes('--retain'))assert.equal(vm.runInContext('draft.rows[0].hollow',c.context),1,'Language redraw must keep edited chat annotations');
 vm.runInContext('draft.rows[0].hollow=1',c.context);
 vm.runInContext("render=()=>{};switchTab=()=>{};act=async(a,p)=>{globalThis.captured=p;return {conversation:'C',project:'P'}};",c.context);
 await vm.runInContext("sendChat('按未知标签导入')",c.context);
 assert.equal(c.context.captured.event,'annotation_save','Retain-unknown must submit current visible annotation rows');
 assert.equal(c.context.captured.inputs.rows[0].hollow,1);
 let restored=false;c.doc.querySelector('#intake').appendChild=()=>{restored=true;};
 vm.runInContext('restorePanelAnnotation()',c.context);assert.equal(restored,true,'Auxiliary annotation controls must return to the visible data panel');
 vm.runInContext("state.jobs=[{id:'J',round:'R',method:'standard',status:'queued'}];state.conversations[0].card={kind:'status',round:'R',jobs:[{id:'J',method:'standard',status:'queued'}]};chatCardSignature='';renderChat();",c.context);
 assert.match(c.doc.querySelector('#chatCard').innerHTML,/queued/);
 vm.runInContext("state.jobs[0].status='running';renderChat();",c.context);
 assert.match(c.doc.querySelector('#chatCard').innerHTML,/running/,'Automatic polling must update chat job statuses without a human event');
 const auditErrors=[];
 async function audit(name,fn){try{await fn();}catch(e){auditErrors.push(name+': '+e.message);}}
 await audit('decision cards bind their own round',async()=>{
  const d=harness();vm.runInContext(fs.readFileSync(path.join(__dirname,'../facade_next/static/chat.js'),'utf8'),d.context);
  const html=vm.runInContext("cardHTML({kind:'comparison',round:'old_round',jobs:[],can_hold:true})",d.context);
  assert.match(html,/data-round="old_round"/);
 });
 await audit('empty HNM shows evaluated parent metrics',async()=>{
  const d=harness();vm.runInContext(fs.readFileSync(path.join(__dirname,'../facade_next/static/chat.js'),'utf8'),d.context);
  d.doc.querySelector('#chatCandidate').value='empty';
  vm.runInContext("state={classes:['hollow'],jobs:[]};var pm={macro_AP:.625,AP_defined_classes:1,classes:{hollow:{AP:.625,precision:.5,recall:.75,F1:.6,FP:2,FN:1}},images:4};renderChatComparison({jobs:[{id:'empty',result:{status:'skip_empty_pool',comparison:{parent:{merged:pm,scenes:{old:pm}},candidate:null}}}]});",d.context);
  assert.match(d.doc.querySelector('#chatComparison').innerHTML||'',/0.625/);
 });
 await audit('closed comparison has no new adoption form',async()=>{
  const d=harness();vm.runInContext(fs.readFileSync(path.join(__dirname,'../facade_next/static/chat.js'),'utf8'),d.context);
  const html=vm.runInContext("cardHTML({kind:'comparison',round:'closed',closed:true,jobs:[],can_hold:true})",d.context);
  assert.doesNotMatch(html,/data-chat-form="propose_decision"/);
 });
 await audit('accepted chat waits for fresh state before another mutation',async()=>{
  const d=harness();vm.runInContext(fs.readFileSync(path.join(__dirname,'../facade_next/static/chat.js'),'utf8'),d.context);
  d.context.posts=[];
  vm.runInContext("state={projects:[{id:'old',name:'old',active:null,train_pool:[],eval_pool:[]}],versions:[],datasets:[],conversations:[{id:'oldC',project:'old',messages:[]}]};projectId='old';chatFresh=true;switchTab=()=>{};render=()=>{if(!state.projects.some(p=>p.id===projectId))projectId='old'};refresh=async()=>{throw Error('offline')};act=async(a,p)=>{posts.push(p);return {project:'new',conversation:'newC',message_count:2}};",d.context);
  await vm.runInContext("sendChat('create project new')",d.context);
  await vm.runInContext("sendChat('Train /new/photos')",d.context).catch(()=>{});
  assert.equal(d.context.posts.length,1,'Second chat must not attach new-project data to an old project after refresh failure');
 });
 assert.deepEqual(auditErrors,[]);
 console.log('UI behavior tests passed (10): original six plus round binding, HNM parent display, closed decisions, stale-state mutation guard');
})().catch(e=>{console.error(e);process.exitCode=1;});
