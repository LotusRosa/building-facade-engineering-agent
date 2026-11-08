const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const nodes=new Map(),listeners={};
const button={disabled:false,dataset:{}};
const doc={documentElement:{},body:{classList:{toggle(){},remove(){}}},querySelector(s){if(!nodes.has(s))nodes.set(s,{value:'',textContent:'',innerHTML:'',hidden:false,children:[],options:[],selectedOptions:[],dataset:{},setAttribute(){},appendChild(){},querySelectorAll:q=>q==='button'?[button]:[],classList:{toggle(){},remove(){}}});return nodes.get(s)},querySelectorAll:()=>[],addEventListener(t,fn){(listeners[t]??=[]).push(fn)}};
class Data {constructor(form){this.entries=form.entries||[];}[Symbol.iterator](){return this.entries[Symbol.iterator]();}getAll(k){return this.entries.filter(([n])=>n===k).map(([,v])=>v);}}
const ctx=vm.createContext({document:doc,console,FormData:Data,structuredClone,crypto:{randomUUID:()=>''},setInterval(){},setTimeout(){},fetch:()=>new Promise(()=>{})});
for(const name of ['app','chat','workspace'])vm.runInContext(fs.readFileSync('facade_next/static/'+name+'.js','utf8'),ctx);
vm.runInContext("state={projects:[{id:'P',name:'test',active:null,train_pool:[],val_pool:[],eval_pool:[]}],versions:[],rounds:[{id:'RI',kind:'initial',project:'P'}],jobs:[],datasets:[],conversations:[{id:'C',project:'P',context:{},messages:[]}],tools:['core','standard','asl','sam','distill','hnm','hem','head_only','ewc_replay'],backbones:['convnext_tiny']};projectId='P';conversationId='C';lang='en';",ctx);
const render=()=>vm.runInContext("cardHTML({kind:'configure',selected:{},options:{tools:[]}})",ctx);
const input=(html,id)=>html.match(new RegExp('<input[^>]*value="'+id+'"[^>]*>'))?.[0]||'';
const initial=render();
assert.match(initial,/Core/);
assert.match(input(initial,'core'),/checked/);
assert.match(input(initial,'core'),/disabled/);
for(const id of ['standard','asl','sam','distill','hnm','hem','head_only','ewc_replay'])assert.match(input(initial,id),/disabled/);
assert.equal(vm.runInContext("methodName('standard',{round:'RI'})",ctx),'Core · Initial modeling');

vm.runInContext("state.projects[0].active='V';state.versions=[{id:'V',contract:{backbone:'convnext_tiny'}}];",ctx);
const update=vm.runInContext("cardHTML({kind:'configure',selected:{},options:{}})",ctx);
assert.match(input(update,'core'),/disabled/);assert(!input(update,'core').includes('checked'));
assert.match(input(update,'standard'),/checked/);assert(!input(update,'standard').includes('disabled'));
const inherited=vm.runInContext("cardHTML({kind:'configure',selected:{},options:{tools:['core']}})",ctx);
assert(!input(inherited,'core').includes('checked'),'Inherited Core must not remain selected in maintenance');
assert.match(input(inherited,'standard'),/checked/);
vm.runInContext("state.rounds[0].status='decided';state.conversations[0].context={options:{tools:['core']}};",ctx);
assert.match(vm.runInContext('projectParameterHTML(state.projects[0])',ctx),/Standard · Adaptation/);
const comparison=vm.runInContext("cardHTML({kind:'comparison',round:'RI',jobs:[{id:'J',method:'standard',status:'succeeded',result:{status:'ready'}}]})",ctx);
assert.match(comparison,/>Core · Initial modeling<\/option>/);
// The auxiliary form is initialized once per project, but must react to adoption.
const toolInputs=['core','standard','sam'].map(value=>({value,checked:value==='core',disabled:value==='core'}));
ctx.toolInputs=toolInputs;
vm.runInContext("document.querySelectorAll=s=>s.startsWith('#tools')?toolInputs:[];projectFormBoundId='P';projectFormBoundActive=null;initializeWorkspaceForm();",ctx);
assert.equal(toolInputs.find(x=>x.value==='standard').checked,true);
toolInputs.find(x=>x.value==='standard').checked=false;
toolInputs.find(x=>x.value==='sam').checked=true;
vm.runInContext('initializeWorkspaceForm()',ctx);
assert.equal(toolInputs.find(x=>x.value==='sam').checked,true,'A valid maintenance draft must survive polling');
assert.equal(vm.runInContext("methodName('standard',{round:'RI'})",ctx),'Core · Initial modeling','An adopted project must not rename its old initial candidate to maintenance');

vm.runInContext("state.projects[0].active=null;sendChat=async(text,event,inputs)=>{globalThis.submitted={text,event,inputs};};",ctx);
const form={dataset:{chatForm:'configure'},entries:[['seed','20260921'],['initialization','cached_imagenet']],querySelector:()=>({value:'convnext_tiny'}),querySelectorAll:()=>[]};
listeners.submit[0]({target:form,preventDefault(){}});
assert.equal(ctx.submitted.event,'configure');assert.deepEqual(Array.from(ctx.submitted.inputs.tools),['core']);
vm.runInContext("state.projects[0].active='V';",ctx);ctx.submitted=null;
assert.doesNotThrow(()=>listeners.submit[0]({target:form,preventDefault(){}}));
assert.equal(ctx.submitted,null);assert.match(doc.querySelector('#notice').textContent,/maintenance method/);
vm.runInContext("state.projects[0].active=null;",ctx);

(async()=>{
 // Restore the production send function to exercise the actual busy lifecycle.
 vm.runInContext(fs.readFileSync('facade_next/static/chat.js','utf8'),ctx);
 let release,calls=0;
 ctx.gated=()=>{calls++;return new Promise(resolve=>release=resolve);};
 vm.runInContext("act=gated;render=()=>{};switchTab=()=>{};conversationId='C';projectId='P';",ctx);
 const first=vm.runInContext("sendChat('confirm','confirm',{token:'exact-confirmation'})",ctx);
 assert.equal(button.disabled,true);
 const duplicate=await vm.runInContext("sendChat('confirm','confirm',{token:'exact-confirmation'})",ctx);
 assert.equal(duplicate,null);assert.equal(calls,1);
 release({conversation:'C',project:'P',message_count:0});await first;
 assert.equal(button.disabled,false);
 console.log('Core and pending UI checks passed (5): initial mandatory Core, maintenance entries, legacy stage labels, initial empty serialization, duplicate confirmation guard');
})().catch(e=>{console.error(e);process.exitCode=1});
