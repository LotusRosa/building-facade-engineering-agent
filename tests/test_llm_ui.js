const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const nodes={};for(const id of ['llmForm','llmStatus','llmProvider','llmBase','llmModel','llmKey','llmSave','llmTest'])nodes['#'+id]={value:'',textContent:'',disabled:false,required:false};
let requests=[],notices=[];
const ctx={state:{csrf:'csrf',llm:{enabled:false,provider:'off',model:'',base_url:'',key_configured:false}},
 $:id=>nodes[id],tr:(zh,en)=>en,handle:async f=>f(),notice:x=>notices.push(typeof x==='function'?x():x),refresh:async()=>{},
 fetch:async(url,opt)=>{requests.push({url,...opt});return {ok:true,json:async()=>JSON.parse(opt.body).provider==='off'?{enabled:false,provider:'off',key_configured:false}:{enabled:true,provider:'deepseek',model:'fake',base_url:'https://api.deepseek.com',key_configured:true}};}};
vm.createContext(ctx);vm.runInContext(fs.readFileSync('facade_next/static/llm.js','utf8'),ctx);
(async()=>{
 ctx.renderLLMStatus();nodes['#llmProvider'].value='deepseek';nodes['#llmProvider'].onchange();
 nodes['#llmModel'].value='typed-model';nodes['#llmKey'].value='fake-secret';
 ctx.renderLLMStatus();assert.equal(nodes['#llmModel'].value,'typed-model');assert.equal(nodes['#llmKey'].value,'fake-secret');
 await nodes['#llmForm'].onsubmit({preventDefault(){}});
 assert.equal(requests.length,1);assert.equal(requests[0].url,'/api/llm/config');
 assert.equal(JSON.parse(requests[0].body).api_key,'fake-secret');assert.equal(nodes['#llmKey'].value,'');
 assert(!nodes['#llmStatus'].textContent.includes('fake-secret'));
 assert.equal(requests[0].headers['X-Facade-CSRF'],'csrf');
 await nodes['#llmTest'].onclick();assert.equal(requests[1].url,'/api/llm/test');
 nodes['#llmProvider'].value='openai';nodes['#llmKey'].value='old';nodes['#llmProvider'].onchange();
 assert.equal(nodes['#llmKey'].value,'');assert.equal(nodes['#llmBase'].value,'https://api.openai.com/v1');
 nodes['#llmProvider'].value='off';nodes['#llmProvider'].onchange();await nodes['#llmForm'].onsubmit({preventDefault(){}});
 assert(notices.at(-1).includes('disabled'));assert(nodes['#llmTest'].disabled);
 assert(!fs.readFileSync('facade_next/static/llm.js','utf8').includes('localStorage'));
 console.log('LLM UI tests passed (5): draft preservation, key clearing, CSRF, explicit connection test, disabled-state message');
})().catch(e=>{console.error(e);process.exitCode=1});
