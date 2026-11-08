// Query routing must not depend on a provider's generated tool arguments.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const nodes=new Map();
const document={documentElement:{},querySelector(s){if(!nodes.has(s))nodes.set(s,{value:'',textContent:'',innerHTML:'',children:[],options:[],selectedOptions:[],querySelectorAll:()=>[],classList:{toggle(){}}});return nodes.get(s);},querySelectorAll:()=>[],addEventListener(){}};
const ctx=vm.createContext({document,console,structuredClone,crypto:{randomUUID:()=>''},setInterval(){},setTimeout(){},fetch:()=>new Promise(()=>{})});
for(const name of ['app','chat'])vm.runInContext(fs.readFileSync('facade_next/static/'+name+'.js','utf8'),ctx);
vm.runInContext("state={projects:[{id:'p',active:null,train_pool:[],eval_pool:[]}],versions:[],datasets:[],jobs:[],rounds:[{id:'initial',project:'p',kind:'initial'}],conversations:[{id:'c',project:'p',messages:[]}]};projectId='p';conversationId='c';render=()=>{};switchTab=()=>{};act=async(action,payload)=>{globalThis.sent=payload;return {conversation:'c',project:'p'}}",ctx);
(async()=>{
 for(const text of ['Compare results','View evaluation results','查看评估结果','Please show results.']){
  await vm.runInContext('sendChat('+JSON.stringify(text)+')',ctx);
  assert.equal(ctx.sent.event,'compare',text+' must use the local results path');assert.equal(ctx.sent.text,text);
 }
 for(const text of ['status','Show status','查看状态','查看训练进度']){
  await vm.runInContext('sendChat('+JSON.stringify(text)+')',ctx);assert.equal(ctx.sent.event,'status');
 }
 for(const text of ['Do not compare results','Compare results and adopt standard','训练数据在 /data/results','why is recall low?']){
  await vm.runInContext('sendChat('+JSON.stringify(text)+')',ctx);assert.equal(ctx.sent.event,null,'Other requests must retain normal interpretation');
 }
 await vm.runInContext("sendChat('Confirm the listed action','confirm',{token:'current-token'})",ctx);assert.equal(ctx.sent.event,'confirm');assert.equal(ctx.sent.inputs.token,'current-token');
 vm.runInContext("lang='en';state.projects[0].active='adopted_initial'",ctx);
 const initial=vm.runInContext("cardHTML({kind:'comparison',round:'initial',jobs:[],can_hold:false})",ctx);
 assert.match(initial,/Initial model evaluation/);assert.doesNotMatch(initial,/Maintenance model comparison/);
 vm.runInContext("state.rounds.push({id:'update',project:'p',kind:'update'})",ctx);
 const update=vm.runInContext("cardHTML({kind:'comparison',round:'update',jobs:[],can_hold:false,closed:true})",ctx);
 assert.match(update,/Maintenance model comparison/);
 console.log('Result-query UI tests passed (4): local result/status routing, mixed-intent preservation, confirmation preservation, stage-specific labels');
})().catch(e=>{console.error(e);process.exitCode=1});
