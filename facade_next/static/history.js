/* Saved records inside the original chat Agent; no workflow actions. */
var AgentHistory=(function(){
'use strict';
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text=(lang,zh,en)=>lang==='zh'?zh:en;
const missing=lang=>text(lang,'未记录','Not recorded');
const number=value=>typeof value==='number'&&Number.isFinite(value)?(Number.isInteger(value)?String(value):value.toFixed(4)):'—';
const methodLabels={core:['Core · 初始建模','Core · Initial modeling'],standard:['Standard · 适应更新','Standard · Adaptation'],asl:['非对称损失（ASL）','Asymmetric loss (ASL)'],sam:['锐度感知优化（SAM）','Sharpness-aware optimization (SAM)'],distill:['历史知识蒸馏','Historical knowledge distillation'],hnm:['困难负样本挖掘','Hard negative mining'],hem:['困难样本重加权','Hard example reweighting'],head_only:['仅更新分类头','Classifier head update only'],ewc_replay:['抗遗忘约束与历史回放（EWC）','EWC constraint with replay']};
function methodLabel(method,kind,lang){const id=method==='standard'&&kind==='initial'?'core':method;return methodLabels[id]?.[lang==='zh'?0:1]||method||missing(lang);}
function fields(rows,lang){return '<dl class="fields">'+rows.map(([key,value])=>`<div><dt>${escapeHTML(key)}</dt><dd>${escapeHTML(value??missing(lang))}</dd></div>`).join('')+'</dl>';}
function table(headers,rows){return '<div class="table-scroll"><table><thead><tr>'+headers.map(h=>'<th>'+escapeHTML(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(v=>'<td>'+escapeHTML(v??'—')+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function renderPanels(data,projectId,roundId,jobId,lang='en'){
 const p=data.projects.find(p=>p.id===projectId),r=data.rounds.find(r=>r.id===roundId&&r.project===projectId),j=data.jobs.find(j=>j.id===jobId&&j.project===projectId&&j.round===r?.id);
 if(!p||!r)return {configuration:'<p>'+text(lang,'请选择该项目已保存的轮次。','Select a saved round in this project.')+'</p>',comparison:'',decisions:''};
 const config=j?.config||r.config||{},protocol=config.protocol||{},kind=config.kind;
 const identity=`<div class="record-identity"><span>${escapeHTML(p.name)}</span><span>${escapeHTML(text(lang,'轮次 ','Round ')+(r.number??'—'))} · ${escapeHTML(r.id)}</span>${j?'<span>'+escapeHTML(j.id)+'</span>':''}</div>`;
 const heading=(letter,title)=>`<div class="sheet-heading"><span class="panel-letter">(${letter})</span><h1>${title}</h1><span class="readonly">${text(lang,'已保存 · 只读','Saved · Read only')}</span></div>${identity}`;
 const counts=config.counts||{},scenes=config.scenes?.evaluation||{},submitted=Object.entries(config.datasets||{}).flatMap(([role,subsets])=>subsets.map(ds=>[role,ds.name||ds.id,ds.count??'—']));
 let configuration=heading('a',text(lang,'任务配置','Task configuration'))+fields([
  [text(lang,'建模阶段','Stage'),kind==='initial'?text(lang,'初始建模','Initial modeling'):kind==='update'?text(lang,'适应更新','Adaptation'):null],
  [text(lang,'训练入口','Training entry'),methodLabel(j?.method,kind,lang)],['Backbone',config.backbone],['Seed',config.seed],
  [text(lang,'权重来源','Weight source'),config.parent?text(lang,'从记录的父模型继续：','Continue from recorded parent: ')+config.parent:kind==='initial'&&config.initialization==='cached_imagenet'?text(lang,'ImageNet 预训练','ImageNet pretrained'):null],
  [text(lang,'任务状态','Job status'),j?.status||r.status],['Train',counts.train],
  [text(lang,'历史 Train 池','Historical Train pool'),counts.history],['Val',counts.val],
  [kind==='initial'?'S0':text(lang,'累计验收集','Cumulative acceptance'),counts.evaluation],
  [text(lang,'最大训练轮数','Epoch budget'),protocol.epochs],[text(lang,'学习率','Learning rate'),protocol.lr],
  [text(lang,'有效批次大小','Effective batch'),protocol.effective_batch],[text(lang,'微批次大小','Micro batch'),protocol.micro_batch],
  [text(lang,'检查点选择','Checkpoint selection'),protocol.checkpoint_selection],
  [text(lang,'训练／评估精度','Train / evaluation precision'),protocol.training_precision&&protocol.evaluation_precision?protocol.training_precision+' / '+protocol.evaluation_precision:null]
 ],lang);
 if(Object.keys(scenes).length)configuration+='<h2>'+text(lang,'保存的验收场景','Saved acceptance scenes')+'</h2>'+table([text(lang,'场景','Scene'),text(lang,'照片数','Photos')],Object.entries(scenes));
 if(submitted.length)configuration+='<h2>'+text(lang,'本轮提交的数据子集','Subsets submitted for this round')+'</h2>'+table([text(lang,'用途','Role'),text(lang,'子集名称','Subset'),text(lang,'照片数','Photos')],submitted);
 configuration+=`<p class="provenance">${text(lang,'来源：','Source: ')}${escapeHTML(config.source||missing(lang))} · ${text(lang,'角色摘要：','Role digest: ')}${escapeHTML(config.roles_digest||missing(lang))}</p>`;
 const c=j?.result?.comparison,skipped=j?.result?.status==='skip_empty_pool';
 let comparison=heading('b',kind==='initial'?text(lang,'初始模型评估','Initial model evaluation'):text(lang,'结果比较','Model comparison'));
 comparison+=`<p class="context">${text(lang,'原任务父模型：','Original task parent: ')}${escapeHTML(config.parent||(kind==='initial'?text(lang,'无','None'):missing(lang)))} · ${escapeHTML(methodLabel(j?.method,kind,lang))}</p>`;
 if(skipped)comparison+='<p class="callout">'+text(lang,'未生成更新权重；以下仅展示保存的父模型评价。','No updated weights; the following is the saved parent evaluation only.')+'</p>';
 if(!c?.candidate&&!c?.parent)comparison+='<p class="callout">'+text(lang,'没有保存的评估结果；查看记录不会运行评价。','No saved evaluation is available. Viewing records does not run evaluation.')+'</p>';
 else{
  const groups=[['merged',c.candidate?.merged,c.parent?.merged],...Array.from(new Set([...Object.keys(c.candidate?.scenes||{}),...Object.keys(c.parent?.scenes||{})])).map(scene=>[scene,c.candidate?.scenes?.[scene],c.parent?.scenes?.[scene]])];
  for(const [scene,candidate,parent] of groups){
   const rows=[];if(parent)rows.push([text(lang,'更新前模型','Parent'),number(parent.macro_AP)]);if(candidate)rows.push([text(lang,'候选模型','Candidate'),number(candidate.macro_AP)]);
   if(typeof parent?.macro_AP==='number'&&typeof candidate?.macro_AP==='number')rows.push(['Δ macro AP',number(candidate.macro_AP-parent.macro_AP)]);
   comparison+='<h2>'+escapeHTML(scene==='merged'?text(lang,'合并验收集','Merged acceptance'):scene)+'</h2>'+table([text(lang,'模型','Model'),'Macro AP'],rows);
   const classes=Array.from(new Set([...Object.keys(parent?.classes||{}),...Object.keys(candidate?.classes||{})]));
   const classRows=classes.flatMap(name=>[['Parent',parent?.classes?.[name]],['Candidate',candidate?.classes?.[name]]].filter(([,values])=>values).map(([label,values])=>[name,label,...['AP','precision','recall','F1','FP','FN'].map(key=>number(values[key]))]));
   if(classRows.length)comparison+=table([text(lang,'类别','Class'),text(lang,'模型','Model'),'AP','Precision','Recall','F1','FP','FN'],classRows);
  }
 }
 const cost=j?.result?.cost||{},costValue=value=>typeof value==='number'&&Number.isFinite(value)?value:missing(lang);
 const start=Date.parse(j?.started),end=Date.parse(j?.finished),elapsed=Number.isFinite(start)&&Number.isFinite(end)&&end>=start?(end-start)/1000:null;
 comparison+='<h2>'+text(lang,'保存的计算成本','Recorded compute cost')+'</h2>'+fields([
  [text(lang,'任务墙钟时间（秒，认领至完成）','Job wall time (s, claim to finish)'),costValue(elapsed)],
  [text(lang,'回执总执行时间（秒）','Receipt execution total (s)'),costValue(cost.total_seconds)],
  [text(lang,'训练及 Val 时间（秒）','Training and Val time (s)'),costValue(cost.training_and_val_seconds)],
  [text(lang,'峰值显存（GiB）','Peak memory (GiB)'),typeof cost.peak_memory_bytes==='number'?costValue(cost.peak_memory_bytes/1024**3):missing(lang)],
  [text(lang,'优化器步数','Optimizer steps'),costValue(cost.optimizer_steps)],
  [text(lang,'人工主动时间（秒）','Human active time (s)'),costValue(cost.human_active_seconds)]
 ],lang);
 comparison+='<p class="provenance">'+text(lang,'来源：已保存的评估回执和任务起止时间；未重新评价。任务墙钟时间包含初始化及评价，不等同于纯 GPU 训练时间。','Source: saved evaluation receipt and job timestamps; no evaluation was rerun. Job wall time includes initialization/evaluation and is not pure GPU training time.')+'</p>';
 const choice=value=>value==='adopt'?text(lang,'采用','Adopt'):value==='hold'?text(lang,'保留父模型（Hold）','Hold'):missing(lang);
 const rounds=data.rounds.filter(x=>x.project===projectId&&x.decision),versions=data.versions.filter(x=>x.project===projectId),rollbacks=(data.rollbacks||[]).filter(x=>x.project===projectId);
 const effective=versions.find(v=>v.id===r.active);
 let decisions=heading('c',text(lang,'版本与决策记录','Version and decision records'))+fields([
  [text(lang,'该轮实际决定','Recorded round decision'),choice(r.decision)],[text(lang,'决定时间','Decision time'),r.decided],
  [text(lang,'当时生效版本','Active version after that decision'),r.active],[text(lang,'当时选中的候选','Selected candidate at that time'),r.selected_job||(r.decision==='hold'?text(lang,'未采用候选（Hold）','None (Hold)'):missing(lang))],
  [text(lang,'生效版本所属轮次','Effective version round'),effective?.round],[text(lang,'生效版本的父版本','Effective version parent'),effective&&Object.hasOwn(effective,'parent')?(effective.parent||text(lang,'无','None')):missing(lang)],
  [text(lang,'工程师理由','Engineer reason'),r.reason]
 ],lang);
 decisions+='<h2>'+text(lang,'已保存的轮次决定','Saved round decisions')+'</h2>'+table([text(lang,'轮次','Round'),text(lang,'决定','Decision'),text(lang,'生效版本','Resulting version'),text(lang,'时间','Time'),text(lang,'理由','Reason')],rounds.map(x=>[x.number,choice(x.decision),x.active,x.decided,x.reason]));
 decisions+='<h2>'+text(lang,'实际采用的版本','Adopted versions')+'</h2>'+table(['Version','Round','Parent','Backbone',text(lang,'采用时间','Adopted'),text(lang,'理由','Reason')],versions.map(v=>[v.id,v.round,v.parent||'—',v.backbone,v.adopted,v.reason]));
 if(rollbacks.length)decisions+='<h2>'+text(lang,'已记录回滚','Recorded rollbacks')+'</h2>'+table(['From','To','UTC',text(lang,'理由','Reason')],rollbacks.map(v=>[v.from_version??v.from,v.to_version??v.to,v.utc||v.created,v.reason]));
 decisions+='<p class="provenance">'+text(lang,'仅展示当时保存的决定。Hold 不创建新采用版本；本页不会再次执行采纳或回滚。','These are historical decisions. Hold creates no adopted version; this page does not repeat adoption or rollback.')+'</p>';
 return {configuration,comparison,decisions};
}

function projectSnapshot(raw){
 const datasets=new Map((raw.datasets||[]).map(x=>[x.id,x]));
 const rounds=new Map((raw.rounds||[]).map(x=>[x.id,x]));
 function config(round,job={}){
  const frozen=job.spec&&typeof job.spec==='object'&&Object.keys(job.spec).length?job.spec:null,source=frozen||round,roles=source.roles||{};
  const counts={},scenes={};
  for(const role of ['train','history','val','evaluation']){
   const rows=roles[role];counts[role]=Array.isArray(rows)&&rows.every(row=>row&&typeof row==='object')?rows.length:null;scenes[role]={};
   if(Array.isArray(rows))for(const row of rows)if(row&&typeof row==='object'){const name=String(row.scene||'(not recorded)');Object.defineProperty(scenes[role],name,{value:(Object.hasOwn(scenes[role],name)?scenes[role][name]:0)+1,writable:true,enumerable:true,configurable:true});}
  }
  const submitted={};for(const [role,ids] of Object.entries(round.datasets||{}))if(Array.isArray(ids))submitted[role]=ids.map(id=>datasets.get(id)).filter(ds=>ds&&ds.project===round.project).map(ds=>({id:ds.id,name:ds.name,count:ds.count}));
  return {...Object.fromEntries(['kind','seed','backbone','initialization','roles_digest'].map(key=>[key,source[key]])),parent:typeof source.parent==='object'?source.parent?.id:source.parent,protocol:source.protocol||{},counts,scenes,datasets:submitted,source:frozen?'job.spec':'round'};
 }
 return {projects:raw.projects||[],rounds:[...rounds.values()].map(r=>({...r,config:config(r)})),jobs:(raw.jobs||[]).filter(j=>rounds.get(j.round)?.project===j.project).map(j=>({...j,config:config(rounds.get(j.round),j)})),versions:(raw.versions||[]).map(v=>({...v,backbone:v.contract?.backbone})),rollbacks:raw.rollbacks||[]};
}

let owner=null,currentPanel='configuration',previousHidden=null,lastHTML='';
const choices=new Map();
function close(){
 const mount=document.querySelector('#savedTaskHistory');if(mount)mount.hidden=true;
 if(previousHidden){for(const [id,hidden] of previousHidden){const element=document.querySelector(id);if(element)element.hidden=hidden;}}
 owner=null;previousHidden=null;lastHTML='';
}
function open(){
 if((typeof chatRequestPending!=='undefined'&&chatRequestPending)||(typeof chatPendingSelection!=='undefined'&&chatPendingSelection))throw Error(tr('请等待当前消息处理完成。','Wait for the current message to finish.'));
 if(!projectId)throw Error(tr('请先选择原来的项目。','Select the original project first.'));
 if(owner!==projectId){close();owner=projectId;previousHidden=['#chatCard','#chatAnnotationSlot'].map(id=>[id,!!document.querySelector(id)?.hidden]);}
 switchTab('conversation');render();
 const scroll=document.querySelector('.chat-scroll');if(scroll)scroll.scrollTop=scroll.scrollHeight;
}
function render(){
 if(!owner)return;
 if(owner!==projectId){close();return;}
 const data=projectSnapshot(state),rounds=data.rounds.filter(r=>r.project===owner).sort((a,b)=>(b.number||0)-(a.number||0)),choice=choices.get(owner)||{};
 if(!rounds.some(r=>r.id===choice.round))choice.round=rounds.find(r=>data.jobs.some(j=>j.round===r.id&&j.status==='succeeded'&&j.result))?.id||rounds[0]?.id||'';
 const jobs=data.jobs.filter(j=>j.project===owner&&j.round===choice.round);
 if(!jobs.some(j=>j.id===choice.job))choice.job=jobs.find(j=>j.status==='succeeded'&&j.result)?.id||jobs[0]?.id||'';
 choices.set(owner,choice);
 const panels=renderPanels(data,owner,choice.round,choice.job,lang),labels={configuration:tr('任务配置','Task configuration'),comparison:tr('结果比较','Model comparison'),decisions:tr('版本记录','Version records')};
 const options=(items,selected,label)=>items.map(item=>`<option value="${escapeHTML(item.id)}" ${item.id===selected?'selected':''}>${escapeHTML(label(item))}</option>`).join('');
 const html=`<div class="agent-history-heading"><strong>${tr('已保存任务 · 只读','Saved tasks · Read only')}</strong><button type="button" class="secondary" data-history-close>${tr('返回当前对话','Back to current chat')}</button></div><div class="grid"><label>${tr('已有轮次','Saved round')}<select id="savedHistoryRound">${options(rounds,choice.round,r=>'Round '+r.number+' · '+r.kind+' · '+r.status)}</select></label><label>${tr('保存的候选','Saved candidate')}<select id="savedHistoryJob">${options(jobs,choice.job,j=>methodLabel(j.method,j.config?.kind,lang)+' · '+j.status)}</select></label></div><div class="agent-history-tabs">${Object.entries(labels).map(([id,label])=>`<button type="button" class="${id===currentPanel?'':'secondary'}" data-history-panel="${id}">${label}</button>`).join('')}<button type="button" class="secondary" data-history-export>${tr('导出 Agent 界面 PNG','Export Agent view PNG')}</button></div><div id="agentHistorySheet">${panels[currentPanel]}</div>`;
 const mount=document.querySelector('#savedTaskHistory');mount.hidden=false;if(html!==lastHTML){mount.innerHTML=html;lastHTML=html;}
 document.querySelector('#chatCard').hidden=true;document.querySelector('#chatAnnotationSlot').hidden=true;
 const title=document.querySelector('#workspaceTitle');if(title)title.textContent=tr('已保存任务','Saved tasks');
}
async function exportAgent(){
 const root=document.querySelector('.app-shell'),width=Math.max(1280,Math.ceil(root.getBoundingClientRect().width)),clone=root.cloneNode(true);
 const exportName={configuration:'agent_task_configuration.png',comparison:'agent_candidate_evaluation.png',decisions:'agent_version_record.png'}[currentPanel];
 clone.classList.add('agent-capture-root');
 // Keep the actual Agent frame, with this saved card; exclude unrelated messages.
 for(const selector of ['#chatMessages','#chatCard','#chatAnnotationSlot','.model-connection-body','#mode','#chatForm','.composer-note'])clone.querySelector(selector)?.remove();
 clone.querySelectorAll('.workspace-project:not(.active),.conversation-row:not(.selected)').forEach(element=>element.remove());
 clone.querySelector('#language')?.querySelectorAll('option').forEach(option=>{if(option.value===lang)option.setAttribute('selected','');else option.removeAttribute('selected');});
 clone.querySelector('#savedTaskHistory')?.removeAttribute('hidden');
 clone.querySelectorAll('[data-history-export],[data-history-close]').forEach(button=>button.remove());
 const css=await Promise.all(['/style.css','/history.css'].map(async url=>{const response=await fetch(url);if(!response.ok)throw Error('Cannot read Agent styles');return response.text();}));
 const measure=document.createElement('div');measure.className='agent-capture-measure';measure.style.width=width+'px';measure.appendChild(clone);document.body.appendChild(measure);
 let height;try{height=Math.ceil(clone.getBoundingClientRect().height)+4;}finally{measure.remove();}
 const scaledHeight=Math.ceil(2600*height/width);
 if(scaledHeight>16000)throw Error(tr('记录过长，请分轮次导出。','This record is too long; export smaller round views.'));
 const markup=new XMLSerializer().serializeToString(clone),svg=`<svg xmlns="http://www.w3.org/2000/svg" width="2600" height="${scaledHeight}" viewBox="0 0 ${width} ${height}"><foreignObject width="${width}" height="${height}"><div xmlns="http://www.w3.org/1999/xhtml" style="width:${width}px"><style>${css.join('\n')}</style>${markup}</div></foreignObject></svg>`;
 const picture=new Image();picture.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg);await new Promise((resolve,reject)=>{picture.onload=resolve;picture.onerror=()=>reject(Error('Agent image export failed'));});
 const canvas=document.createElement('canvas');canvas.width=2600;canvas.height=scaledHeight;const context=canvas.getContext('2d');context.fillStyle='#f3f5f1';context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(picture,0,0,canvas.width,canvas.height);
 const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));if(!blob)throw Error('PNG export failed');const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=exportName;link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
 notice(tr('已导出原 Agent 界面。','Exported the original Agent view.'));
}
if(typeof document!=='undefined'){
 document.addEventListener('change',event=>{if(!owner)return;const choice=choices.get(owner);if(event.target.id==='savedHistoryRound'){choice.round=event.target.value;choice.job='';render();}if(event.target.id==='savedHistoryJob'){choice.job=event.target.value;render();}});
 document.addEventListener('click',event=>{const button=event.target.closest('button');if(!button)return;if(button.hasAttribute('data-history-close')){close();if(typeof renderWorkspace==='function')renderWorkspace();}if(button.dataset.historyPanel){currentPanel=button.dataset.historyPanel;render();}if(button.hasAttribute('data-history-export'))handle(async()=>{button.disabled=true;try{await exportAgent();}finally{button.disabled=false;}});});
 document.addEventListener('submit',event=>{if(event.target.id==='chatForm')close();},true);
}
return {projectSnapshot,renderPanels,open,close,render};
})();
