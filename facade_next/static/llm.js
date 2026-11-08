// Credentials stay in this form until saved to server memory. Never browser storage.
let llmFormInitialized=false;
const llmBases={openai:'https://api.openai.com/v1',deepseek:'https://api.deepseek.com',compatible:'',off:''};
function llmFields(){const enabled=$('#llmProvider').value!=='off';for(const id of ['#llmBase','#llmModel','#llmKey'])$(id).disabled=!enabled;$('#llmModel').required=enabled;}
function renderLLMStatus(){
 const s=state?.llm||{provider:'off',enabled:false};
 $('#llmStatus').textContent=s.enabled?`${s.provider} · ${s.model} · ${tr('设置已保存','Settings saved')}`:(s.error||tr('规则聊天 · LLM 未启用','Rule-based chat · LLM disabled'));
 $('#llmTest').disabled=!s.enabled;
 if(!llmFormInitialized){$('#llmProvider').value=s.provider;$('#llmBase').value=s.base_url||'';$('#llmModel').value=s.model||'';llmFormInitialized=true;llmFields();}
}
async function llmRequest(path,payload){
 const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Facade-CSRF':state.csrf},body:JSON.stringify(payload)});
 const data=await r.json();if(!r.ok)throw Error(data.error||'LLM settings failed');return data;
}
$('#llmProvider').onchange=()=>{$('#llmBase').value=llmBases[$('#llmProvider').value];$('#llmModel').value='';$('#llmKey').value='';llmFields();};
$('#llmForm').onsubmit=e=>{e.preventDefault();return handle(async()=>{
 $('#llmSave').disabled=true;
 try{
  const s=await llmRequest('/api/llm/config',{provider:$('#llmProvider').value,base_url:$('#llmBase').value.trim(),model:$('#llmModel').value.trim(),api_key:$('#llmKey').value});
  $('#llmKey').value='';state.llm=s;renderLLMStatus();
   notice(()=>s.enabled?tr('连接设置已保存到本次服务内存；尚未验证密钥，可点击测试连接。','Settings saved for this service session; use Test connection to verify credentials.'):tr('LLM 已关闭，继续使用规则聊天。','LLM disabled; rule-based chat remains available.'));
 }finally{$('#llmSave').disabled=false;}
});};
$('#llmTest').onclick=()=>handle(async()=>{$('#llmTest').disabled=true;try{await llmRequest('/api/llm/test',{});notice(()=>tr('LLM 接口测试通过。','LLM connection test passed.'));}finally{$('#llmTest').disabled=!state.llm?.enabled;}});
if(state)renderLLMStatus();
