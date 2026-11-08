const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
assert(fs.existsSync('facade_next/static/environment.js'),'First-run environment wizard is not implemented');
const nodes=new Map(),requests=[];
const $=s=>{if(!nodes.has(s))nodes.set(s,{value:'',textContent:'',innerHTML:'',hidden:true,disabled:false,checked:false,querySelectorAll:()=>[]});return nodes.get(s)};
const ctx=vm.createContext({state:{environment:{ready:false,current_prefix:'/usr',default_folder:'/home/test/FacadeAgentEnvironment',operation:{state:'idle'}}},
 $,$$:()=>[],lang:'en',tr:(zh,en)=>ctx.lang==='zh'?zh:en,esc:x=>String(x),handle:async f=>f(),notice(){},translate(){},refresh:async()=>{},
 fetch:async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>({ready:true,profile:{folder:'/chosen',python:'/chosen/.venv/bin/python',gpus:'1'},operation:{state:'succeeded'}})}}});
vm.runInContext(fs.readFileSync('facade_next/static/environment.js','utf8'),ctx);
(async()=>{
 vm.runInContext('renderEnvironment()',ctx);assert.equal($('#environmentWizard').hidden,false);
 $('#envFolder').value='/chosen';$('#chatInput').value='unsent message';
 ctx.state.environment.operation={state:'running',phase:'Creating the training environment',log:'fixture progress'};
 vm.runInContext('renderEnvironment()',ctx);assert.equal($('#envFolder').value,'/chosen');assert.equal($('#chatInput').value,'unsent message');assert.match($('#envLog').textContent,/fixture progress/);
 $('#envGpus').value='1';$('#envMode').value='existing';ctx.state.csrf='csrf';ctx.state.environment.operation={state:'idle'};
 await $('#environmentForm').onsubmit({preventDefault(){}});
 const posted=JSON.parse(requests[0].options.body);assert.equal(posted.folder,'/chosen');assert.equal(posted.mode,'existing');assert.equal(posted.gpus,'1');assert.equal(requests[0].options.headers['X-Facade-CSRF'],'csrf');
 await $('#envLater').onclick();assert.equal($('#environmentWizard').hidden,true);vm.runInContext('renderEnvironment()',ctx);assert.equal($('#environmentWizard').hidden,true);
 ctx.lang='zh';vm.runInContext('renderEnvironment()',ctx);assert.equal($('#envFolder').value,'/chosen');assert.equal($('#chatInput').value,'unsent message');
 ctx.lang='en';ctx.state.environment.operation={state:'running',phase:'Checking Python, CUDA/BF16 and requested ImageNet weights',elapsed_seconds:65,log:'fixture weight download',download:{backbone:'convnext_tiny',status:'downloading',received:10485760,total:20971520,attempt:2,verified:false}};
 vm.runInContext('renderEnvironment()',ctx);assert.equal($('#envDownloadBar').value,.5);assert.match($('#envDownloadText').textContent,/ConvNeXt-Tiny/);assert.match($('#envDownloadText').textContent,/2\/3/);assert.match($('#envElapsed').textContent,/1m/);
 ctx.state.environment.operation.state='failed';ctx.state.environment.operation.error='SSL interrupted';vm.runInContext('renderEnvironment()',ctx);assert.equal($('#envRetryWeights').hidden,false);
 let retried=false;$('#environmentForm').requestSubmit=()=>{retried=true};$('#envMode').value='create';$('#envRetryWeights').onclick();assert.equal($('#envMode').value,'existing');assert.equal(retried,true);
 console.log('Environment UI behavior tests passed (6): first-run, progress/draft preservation, shared setup submission, dismiss/language persistence, real weight bytes/elapsed, reuse-on-weight-failure action');
})().catch(e=>{console.error(e);process.exitCode=1});
