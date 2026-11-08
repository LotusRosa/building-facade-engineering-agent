// Exercise language switching with the shipped HTML and real UI handlers.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.join(__dirname,'../facade_next/static');
function harness(){
 const nodes=new Map(),all=[];
 const doc={title:'',documentElement:{},addEventListener(){},querySelector(s){if(!nodes.has(s))nodes.set(s,element());return nodes.get(s);},querySelectorAll(s){
  const key={'[data-zh]':'zh','[data-placeholder-zh]':'placeholderZh','[data-aria-zh]':'ariaZh'}[s];
  return key?all.filter(e=>Object.hasOwn(e.dataset,key)):[];
 }};
 function element(){return {dataset:{},value:'',placeholder:'',textContent:'',innerHTML:'',className:'',children:[],options:[],selectedOptions:[],querySelectorAll:()=>[],setAttribute(k,v){this[k]=v;},classList:{toggle(){}}};}
 const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
 for(const m of html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/gi)){
  const e=element();
  for(const a of m[2].matchAll(/([\w-]+)="([^"]*)"/g)){
   if(a[1].startsWith('data-'))e.dataset[a[1].slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=a[2];
   else e[a[1]]=a[2];
  }
  if(m[1]==='title')Object.defineProperty(e,'textContent',{get(){return doc.title;},set(v){doc.title=v;}});
  all.push(e);if(e.id)nodes.set('#'+e.id,e);
 }
 const requests=[];
 const ctx=vm.createContext({document:doc,console,structuredClone,crypto:{randomUUID:()=>''},setInterval(){},setTimeout(){},fetch:()=>new Promise(()=>{})});
 vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),ctx);
 vm.runInContext("state={csrf:'test',projects:[],datasets:[],rounds:[],jobs:[],versions:[],events:[],backbones:['convnext_tiny'],tools:['standard'],method_presets:{},llm:{enabled:true,provider:'deepseek',model:'test-model',base_url:'https://api.deepseek.com',key_configured:true}}",ctx);
 ctx.fetch=async(url,opt)=>{requests.push(url);return {ok:true,json:async()=>url==='/api/llm/config'?{enabled:JSON.parse(opt.body).provider!=='off',provider:JSON.parse(opt.body).provider,model:'test-model',key_configured:true}:{status:'ok'}};};
 vm.runInContext(fs.readFileSync(path.join(root,'llm.js'),'utf8'),ctx);
 return {doc,ctx,requests,change(language){doc.querySelector('#language').onchange({target:{value:language}});}};
}
(async()=>{
 const failures=[];
 async function check(name,fn){try{await fn();console.log('PASS '+name);}catch(e){failures.push(name+': '+e.message);}}
 await check('language switch translates prompts while preserving typed chat and model settings',()=>{
  const h=harness();h.doc.querySelector('#chatInput').value='未发送的项目消息';h.doc.querySelector('#llmKey').value='unsaved-fake-key';h.doc.querySelector('#llmModel').value='typed-model';
  h.change('en');
  assert.match(h.doc.querySelector('#chatInput').placeholder,/create project/i);
  assert.doesNotMatch(h.doc.querySelector('#chatInput').placeholder,/[\u3400-\u9fff]/);
  for(const id of ['#llmKey','#llmModel'])assert.doesNotMatch(h.doc.querySelector(id).placeholder,/[\u3400-\u9fff]/);
  assert.equal(h.doc.querySelector('#chatInput').value,'未发送的项目消息');
  assert.equal(h.doc.querySelector('#llmKey').value,'unsaved-fake-key');assert.equal(h.doc.querySelector('#llmModel').value,'typed-model');
  assert.equal(h.doc.title,'building-facade-engineering-agent');
  h.change('zh');assert.match(h.doc.querySelector('#chatInput').placeholder,/创建项目/);assert.equal(h.doc.title,'building-facade-engineering-agent');
 });
 await check('completed connection notice follows language without another API request',async()=>{
  const h=harness();await h.doc.querySelector('#llmTest').onclick();assert.match(h.doc.querySelector('#notice').textContent,/测试通过/);
  h.change('en');assert.match(h.doc.querySelector('#notice').textContent,/connection test passed/i);assert.equal(h.requests.length,1);
  h.change('zh');assert.match(h.doc.querySelector('#notice').textContent,/测试通过/);assert.equal(h.requests.length,1);
 });
 await check('saved and disabled connection notices follow language',async()=>{
  const h=harness();await h.doc.querySelector('#llmForm').onsubmit({preventDefault(){}});h.change('en');assert.match(h.doc.querySelector('#notice').textContent,/settings saved/i);
  h.doc.querySelector('#llmProvider').value='off';await h.doc.querySelector('#llmForm').onsubmit({preventDefault(){}});h.change('zh');assert.match(h.doc.querySelector('#notice').textContent,/LLM 已关闭/);
 });
 await check('a later raw server error stays visible after language changes',async()=>{
  const h=harness();await h.doc.querySelector('#llmTest').onclick();vm.runInContext("notice('provider request failed: 429',true)",h.ctx);h.change('en');
  assert.equal(h.doc.querySelector('#notice').textContent,'provider request failed: 429');assert.equal(h.doc.querySelector('#notice').className,'error');
 });
 assert.deepEqual(failures,[]);console.log('Locale UI behavior tests passed (4).');
})().catch(e=>{console.error(e);process.exitCode=1});
