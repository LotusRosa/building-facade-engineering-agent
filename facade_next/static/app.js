let state=null,lang='zh',projectId='',draft=null,page=0,pending=new Map();
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const tr=(zh,en)=>lang==='zh'?zh:en;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const chosen=s=>[...$(s).selectedOptions].map(x=>x.value);
const project=()=>state?.projects.find(x=>x.id===projectId);
// Display labels mirror names.py; canonical IDs remain form values and audit keys.
const METHOD_LABELS={"core":{"zh":"Core · 初始建模","en":"Core · Initial modeling"},"standard":{"zh":"Standard · 常规维护","en":"Standard · Adaptation"},"asl":{"zh":"非对称损失（ASL）","en":"Asymmetric loss (ASL)"},"sam":{"zh":"锐度感知优化（SAM）","en":"Sharpness-aware optimization (SAM)"},"distill":{"zh":"历史知识蒸馏","en":"Historical knowledge distillation"},"hnm":{"zh":"困难负样本挖掘","en":"Hard negative mining"},"hem":{"zh":"困难样本重加权","en":"Hard example reweighting"},"head_only":{"zh":"仅更新分类头","en":"Classifier head update only"},"ewc_replay":{"zh":"抗遗忘约束与历史回放（EWC）","en":"EWC constraint with replay"}};
const entryId=(id,context=null)=>{const kind=typeof context==='string'?context:context?.kind||context?.spec?.kind||state?.rounds?.find(r=>r.id===context?.round)?.kind;return id==='standard'&&kind==='initial'?'core':id;};
const methodName=(id,context=null)=>METHOD_LABELS[entryId(id,context)]?.[lang]||entryId(id,context);
function inheritedEntryTools(tools,active){if(!active)return ['core'];const raw=tools??['standard'];return raw.includes('core')?(raw.filter(t=>t!=='core').length?raw.filter(t=>t!=='core'):['standard']):raw;}
function selectedEntryTools(form,active){const tools=[...form.querySelectorAll('[name="tool"]:checked')].map(x=>x.value);if(!active)return ['core'];if(!tools.length)throw Error(tr('请至少选择一种维护方法，例如 Standard。','Select at least one maintenance method, such as Standard.'));return tools;}
const stageName=kind=>kind==='initial'?tr('初始建模','Initial modeling'):kind==='update'?tr('模型维护','Model maintenance'):kind;
function resultStage(roundId=null,pid=projectId){const rounds=state?.rounds||[],r=roundId?rounds.find(x=>x.id===roundId):rounds.filter(x=>x.project===pid).at(-1);return r?.kind||((state?.projects||[]).find(p=>p.id===pid)?.active?'update':'initial');}
function resultsTitle(roundId=null){return resultStage(roundId)==='initial'?tr('初始模型评估结果','Initial model evaluation'):tr('维护轮模型对比','Maintenance model comparison');}
function resultsLabel(pid=projectId){return resultStage(null,pid)==='initial'?tr('评估结果','Evaluation'):tr('模型对比','Comparison');}
function jobProgressHTML(job){
 const p=job.progress||{stage:job.status},labels={queued:tr('等待 GPU 队列','Queued'),preparing:tr('准备图像缓存','Preparing image cache'),initializing:tr('准备数据与模型','Preparing data and model'),training:tr('训练与每轮 Val 检查','Training and epoch validation'),validation:tr('\u672c\u8f6e Val \u68c0\u67e5','Epoch Val validation'),calibrating:tr('\u5b8c\u6210\u8bad\u7ec3\u5e76\u6821\u51c6 Val \u9608\u503c','Completing training and Val calibration'),skipped:tr('HNM \u7a7a\u6c60\uff1a\u672c\u8f6e\u672a\u8bad\u7ec3\u65b0\u6743\u91cd','Empty HNM pool: no update training'),evaluation:tr('验收评价','Evaluation'),finalizing:tr('保存结果与校验收据','Saving and verifying results'),completed:tr('训练与评价已完成','Training and evaluation completed'),failed:tr('任务失败，请查看日志','Failed; inspect the log'),cancelled:tr('已取消','Cancelled')};
 const known=typeof p.fraction==='number'&&Number.isFinite(p.fraction),fraction=known?Math.max(0,Math.min(1,p.fraction)):null;
 let detail='';
 if(p.stage==='preparing')detail=`${p.cached_images} / ${p.total_images} ${tr('张图像已有缓存','images cached')}`;
 if(['training','validation'].includes(p.stage))detail=`Epoch ${p.epoch} / ${p.max_epochs} ${tr('（最多，可早停）','(maximum; early stopping may apply)')} · Step ${p.step} / ${p.steps_per_epoch} · ${tr('累计训练步','Training steps')} ${p.completed_steps} / ${p.planned_steps}`;
 if(p.stage==='completed'&&p.completed_epochs)detail=`${tr('实际完成','Completed')} ${p.completed_epochs} ${tr('轮，计划最多','epochs; planned maximum')} ${p.max_epochs}`;
 if(p.stage==='skipped')detail=tr('\u5df2\u5b8c\u6210\u7236\u6a21\u578b\u8bc4\u4ef7\uff0c\u4e0d\u4f1a\u751f\u6210\u66f4\u65b0\u6a21\u578b\u3002','Parent evaluation completed; no updated weights were created.');
 const elapsed=Number.isFinite(p.elapsed_seconds)?`${tr('任务耗时','Elapsed')} ${Math.floor(p.elapsed_seconds/60)}m ${p.elapsed_seconds%60}s`:'';
 return `<div class="job-progress"><div class="job-progress-heading"><strong>${job.method?esc(methodName(job.method,job))+' · ':''}${esc(labels[p.stage]||p.stage||job.status)}</strong><span>${known?Math.round(fraction*100)+'%':''}</span></div><progress max="1" ${known?'value="'+fraction+'"':''} aria-label="${esc(labels[p.stage]||job.status)}"></progress><small>${esc(detail)}${elapsed?' · '+esc(elapsed):''}</small></div>`;
}
let currentNotice=null;
function renderNotice(){if(!currentNotice)return;const {text,error}=currentNotice;$('#notice').textContent=typeof text==='function'?text():text;$('#notice').className=error?'error':'';}
function notice(text,error=false){currentNotice={text,error};renderNotice();}
function translate(){
 $$('[data-zh]').forEach(e=>e.textContent=e.dataset[lang]);
 $$('[data-placeholder-zh]').forEach(e=>e.placeholder=e.dataset[lang==='zh'?'placeholderZh':'placeholderEn']);
 $$('[data-aria-zh]').forEach(e=>e.setAttribute('aria-label',e.dataset[lang==='zh'?'ariaZh':'ariaEn']));
 document.documentElement.lang=lang==='zh'?'zh-CN':'en';renderNotice();
}
async function refresh(){const r=await fetch('/api/state');if(!r.ok)throw Error('State unavailable');state=await r.json();if(typeof applyChatSelection==='function')applyChatSelection();render();}
async function act(action,payload){
 const key=JSON.stringify({action,payload});let op=pending.get(key);
 if(!op){op={id:crypto.randomUUID(),accepted:false,result:null};pending.set(key,op);}
 if(!op.accepted){
  const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json','X-Facade-CSRF':state.csrf},body:JSON.stringify({action,payload,submission_id:op.id})});
  const result=await r.json();if(!r.ok){pending.delete(key);throw Error(result.error);}
  op.accepted=true;op.result=result;
 }
 try{await refresh();pending.delete(key);notice(()=>tr('已保存。','Saved.'));}
 catch(e){notice(()=>tr('提交已保存；状态刷新暂不可用，可重试查询。','Submission saved; state refresh unavailable. Retry inspection.'));}
 return op.result;
}
async function handle(fn){try{await fn();}catch(e){notice(e.message,true);}}
function table(head,rows){return `<div class="tablewrap"><table><thead><tr>${head.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function switchTab(id){if(typeof workspaceView!=='undefined')workspaceView=id;$$('.panel').forEach(e=>e.hidden=e.id!==id);$$('[data-tab]').forEach(e=>e.classList.toggle('selected',e.dataset.tab===id));if(id==='conversation'){const scroll=$('.chat-scroll');if(scroll)scroll.scrollTop=scroll.scrollHeight;}}
function preserveOptions(element,options){const old=new Set(chosen('#'+element.id));element.innerHTML=options;element.querySelectorAll('option').forEach(x=>x.selected=old.has(x.value));}
function render(){
 if(!state)return;
 if(!state.projects.some(p=>p.id===projectId))projectId=state.projects[0]?.id||'';
 $('#project').innerHTML=state.projects.map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join('');$('#project').value=projectId;
 $('#mode').textContent=state.workers_enabled?tr('GPU Worker 已启用','GPU workers enabled'):tr('界面模式 · 不执行模型','UI mode · no model execution');
 $('#settingsMode').textContent=state.workers_enabled?tr('训练执行已启用','Training execution enabled'):tr('训练执行未启用','Training execution disabled');
 const p=project(),ds=state.datasets.filter(x=>x.project===projectId),rounds=state.rounds.filter(x=>x.project===projectId),jobs=state.jobs.filter(x=>x.project===projectId),versions=state.versions.filter(x=>x.project===projectId);
 $('#datasets').innerHTML=table([tr('数据集','Dataset'),tr('照片数','Photos'),'ID','SHA256'],ds.map(d=>[esc(d.name),d.count,`<code>${d.id}</code>`,`<code>${d.digest.slice(0,16)}…</code>`]));
 for(const id of ['trainData','valData','evalData'])preserveOptions($('#'+id),ds.map(d=>`<option value="${d.id}">${esc(d.name)} (${d.count})</option>`).join(''));
 const active=versions.find(v=>v.id===p?.active);$('#initializationSource').textContent=active?tr('从当前在役版本继续更新','Continue from active version'):tr('ImageNet 预训练（本机缓存）','ImageNet pretrained (local cache)');
 const previousBackbone=$('#backbone').value;
 $('#backbone').innerHTML=state.backbones.map(b=>`<option>${b}</option>`).join('');$('#backbone').value=active?.contract.backbone||previousBackbone||'convnext_tiny';$('#backbone').disabled=!!active;
 $('#activeSummary').textContent=active?tr(`在役：${active.id} · ${active.contract.backbone}；历史训练 ${p.train_pool.length}，累计Val ${p.val_pool.length}，历史评价 ${p.eval_pool.length}。`, `Active: ${active.id} · ${active.contract.backbone}; historical train ${p.train_pool.length}, Val ${p.val_pool.length}, evaluation ${p.eval_pool.length}.`):tr('尚无在役模型：创建初始模型，或在服务器显式导入已验证权重与原始角色清单。','No active model: build an initial model or explicitly import validated weights and original role manifests on the server.');
 if(!$('#tools').children.length)$('#tools').innerHTML=state.tools.map(t=>`<label><input type="checkbox" name="tool" value="${t}" ${t===(active?'standard':'core')?'checked':''}><span data-zh="${esc(METHOD_LABELS[t]?.zh||t)}" data-en="${esc(METHOD_LABELS[t]?.en||t)}">${esc(methodName(t))}</span></label>`).join('');
 $$('#tools input').forEach(x=>{x.disabled=!active||x.value==='core';if(!active)x.checked=x.value==='core';else if(x.value==='core')x.checked=false;});
 $('#presetDetails').innerHTML=`<details><summary>${tr('查看训练固定参数','View fixed training parameters')}</summary><p>${tr('Core 使用初始训练协议；Standard 与其余方法使用维护协议。','Core uses the initial protocol; Standard and other methods use the maintenance protocol.')}</p><pre>${esc(JSON.stringify(state.method_presets,null,2))}</pre></details>`;
 $('#rounds').innerHTML=table([tr('轮次','Round'),tr('状态','Status'),tr('角色数：新 / 旧 / Val / 评价','Rows: new / replay / Val / eval'),tr('操作','Action')],rounds.map(r=>[`${r.number} · ${esc(stageName(r.kind))}<small>${r.id}</small>`,esc(r.status),['train','history','val','evaluation'].map(k=>r.roles[k].length).join(' / '),`${r.status==='draft'?`<button data-start="${r.id}">${tr('启动所选候选','Start selected candidates')}</button>`:''}${!['decided','cancelled'].includes(r.status)?` <button class="secondary" data-cancel="${r.id}">${tr('取消本轮','Cancel round')}</button>`:''}`]));
  $('#jobs').innerHTML=table([tr('工具','Tool'),tr('状态与进度','Status and progress'),tr('计算','Compute'),tr('日志 / 错误','Log / error')],jobs.map(j=>[`${esc(methodName(j.method,j))}<small>${j.id}</small>`,esc(j.status)+jobProgressHTML(j),esc(j.device||'—'),`<button data-log="${j.id}" class="secondary">${tr('查看日志','View log')}</button>${j.error?`<details><summary>${tr('错误','Error')}</summary><pre>${esc(j.error)}</pre></details>`:''}${j.status==='running'?`<button data-interrupted="${j.id}" class="secondary">${tr('恢复中断状态','Recover interrupted state')}</button>`:''}`]));
 preserveOptions($('#decisionRound'),[...rounds].reverse().map(r=>`<option value="${r.id}">${r.number} · ${r.status}</option>`).join(''));if(!$('#decisionRound').value&&$('#decisionRound').options.length)$('#decisionRound').selectedIndex=0;renderCandidates();
 $('#versions').innerHTML=table([tr('版本','Version'),tr('主干 / 阈值','Backbone / thresholds'),tr('父版本','Parent'),tr('理由','Reason')],versions.map(v=>[`${v.id===p?.active?'● ':''}<code>${v.id}</code>`,`${esc(v.contract.backbone)}<small>${esc(v.contract.thresholds.join(', '))}</small>`,esc(v.parent||'—'),esc(v.reason)]));
 preserveOptions($('#rollbackVersion'),versions.map(v=>`<option value="${v.id}">${v.id}${v.id===p?.active?' · active':''}</option>`).join(''));
 const events=state.events.filter(e=>e.project===projectId||rounds.some(r=>r.id===e.round));$('#eventCount').textContent=`O=${events.reduce((s,e)=>s+e.O,0)} · K=${events.reduce((s,e)=>s+e.K,0)}`;
 $('#events').innerHTML=table(['UTC',tr('人工动作','Human action'),'O','K',tr('提交ID','Submission ID')],events.slice(-50).reverse().map(e=>[esc(e.utc),esc(e.action),e.O,e.K,`<code>${esc(e.submission_id)}</code>`]));
  $('#timerState').textContent=JSON.stringify(p?.timer||{state:'off'},null,2);if(typeof initializeWorkspaceForm==='function')initializeWorkspaceForm();translate();if(typeof renderChat==='function')renderChat();if(typeof renderLLMStatus==='function')renderLLMStatus();if(typeof renderWorkspace==='function')renderWorkspace();if(typeof renderEnvironment==='function')renderEnvironment();if(typeof AgentHistory!=='undefined')AgentHistory.render();
}
function fmt(v){return v===null||v===undefined?'—':typeof v==='number'?(Number.isInteger(v)?v:v.toFixed(4)):esc(v);}
function renderCandidates(){const rid=$('#decisionRound').value;preserveOptions($('#comparisonJob'),state.jobs.filter(j=>j.round===rid&&j.status==='succeeded').map(j=>`<option value="${j.id}">${esc(methodName(j.method,j))} · ${j.id}</option>`).join(''));if(!$('#comparisonJob').value&&$('#comparisonJob').options.length)$('#comparisonJob').selectedIndex=0;renderComparison();}
function renderComparison(){
 const r=state.rounds.find(x=>x.id===$('#decisionRound').value),job=state.jobs.find(x=>x.id===$('#comparisonJob').value),c=job?.result?.comparison;
  $('#resultsHeading').textContent=resultsTitle(r?.id);
  $('#resultsIntro').textContent=resultStage(r?.id)==='initial'?tr('查看初始模型在 S0 验收集上的评估指标。本轮没有父模型，不做新旧模型对比。','Review the initial model on the S0 acceptance set. This round has no parent-model comparison.'):tr('各模型使用自身的 Val 阈值，比较候选与本轮父模型。历史退化仅作提示，由你决定采用或保留。','Compare the candidate and round parent using each model’s own Val thresholds. Historical loss is advisory; you decide whether to adopt or retain.');
 $('#comparison').innerHTML='';
 const pendingJobs=r&&state.jobs.some(j=>j.round===r.id&&['queued','running'].includes(j.status));
 $('#adopt').disabled=!job||job.result.status!=='ready'||['decided','cancelled'].includes(r?.status)||pendingJobs;
 $('#hold').disabled=!r?.parent||['decided','cancelled'].includes(r?.status)||pendingJobs;
 $('#decisionHint').textContent=tr('训练完成后仍需人工确认；决定后下一轮沿用实际在役模型与阈值。','Training completion requires an engineer decision. The next round uses the actual active weights and thresholds.');
 if(!c)return;
 if(!c.candidate){$('#comparison').textContent=tr('HNM 空池：未创建更新权重，可保留父模型。','HNM empty pool: no updated weights; retain the parent.');return;}
 const groups=[['merged',c.candidate.merged,c.parent?.merged],...Object.entries(c.candidate.scenes).map(([name,m])=>[name,m,c.parent?.scenes[name]])];
 for(const [name,m,old] of groups){const delta=old?.macro_AP!=null&&m.macro_AP!=null?m.macro_AP-old.macro_AP:null;
  $('#comparison').innerHTML+=`<h3>${esc(name)} · macro AP ${fmt(m.macro_AP)} (${m.AP_defined_classes}/3) ${delta!==null?`Δ ${fmt(delta)}`:''}</h3>`+table([tr('类别 / 版本','Class / version'),'AP','Precision','Recall','F1','FP','FN',tr('已知标签','Known labels')],state.classes.flatMap(k=>{const row=(label,x)=>[esc(k)+' · '+label,...['AP','precision','recall','F1','FP','FN','known'].map(n=>fmt(x[n]))];return old?[row(tr('父','parent'),old.classes[k]),row(tr('候选','candidate'),m.classes[k])]:[row(tr('候选','candidate'),m.classes[k])];}));
 }
}
function collectAnnotations(){$$('#photoRows [data-index]').forEach(e=>{const r=draft.rows[Number(e.dataset.index)];e.querySelectorAll('[data-field]').forEach(input=>{r[input.dataset.field]=input.value===''?null:input.value;});});}
function renderPhotos(){const start=page*20;$('#photoRows').innerHTML=draft.rows.slice(start,start+20).map((r,k)=>{const i=start+k;return `<div class="photo-row" data-index="${i}"><img class="photo" loading="lazy" src="/api/photo?draft=${draft.id}&index=${i}" alt="${esc(r.image_id)}"><span>${esc(r.image_id)}</span>${state.classes.map(c=>`<label>${c}<select data-field="${c}">${[['',tr('未知','Unknown')],['0',tr('阴性 0','Negative 0')],['1',tr('阳性 1','Positive 1')]].map(([v,t])=>`<option value="${v}" ${String(r[c]??'')===v?'selected':''}>${t}</option>`).join('')}</select></label>`).join('')}<label>scene<input data-field="scene" value="${esc(r.scene)}"></label><label>group<input data-field="group" value="${esc(r.group)}"></label></div>`;}).join('');$('#page').textContent=`${page+1} / ${Math.ceil(draft.rows.length/20)}`;$('#prev').disabled=page===0;$('#next').disabled=(page+1)*20>=draft.rows.length;}
$('#language').onchange=e=>{if(draft)collectAnnotations();lang=e.target.value;translate();render();if(draft)renderPhotos();};
$('#project').onchange=e=>{if(typeof resetChat==='function')resetChat();projectId=e.target.value;draft=null;$('#annotation').hidden=true;render();};
$$('[data-tab]').forEach(e=>e.onclick=()=>switchTab(e.dataset.tab));
$('#newProject').onsubmit=e=>{e.preventDefault();handle(async()=>{if(chatPendingSelection||chatRequestPending)throw Error(tr('请等上一条消息处理完。','Wait for the previous message.'));const p=await act('create_project',{name:new FormData(e.target).get('name')});selectWorkspaceProject(p.id);startWorkspaceChat();e.target.reset();$('#newProjectDetails').open=false;});};
$('#refresh').onclick=()=>handle(async()=>{if(projectId)await act('inspect',{project:projectId});else await refresh();});
$('#importForm').onsubmit=e=>{e.preventDefault();handle(async()=>{if(!projectId)throw Error(tr('请先创建项目','Create a project first'));const p=Object.fromEntries(new FormData(e.target));p.project=projectId;if(p.manifest){await act('import_dataset',p);}else{if(typeof restorePanelAnnotation==='function')restorePanelAnnotation();draft=await act('scan_folder',{project:projectId,root:p.root});draft.name=p.name;draft.rows.forEach(r=>r.scene=p.name);page=0;$('#annotation').hidden=false;renderPhotos();}});};
$('#prev').onclick=()=>{collectAnnotations();page--;renderPhotos();};$('#next').onclick=()=>{collectAnnotations();page++;renderPhotos();};
$('#saveAnnotation').onclick=()=>handle(async()=>{collectAnnotations();if(typeof saveChatAnnotation==='function'&&chatDraft){await saveChatAnnotation();}else{await act('import_dataset',{project:projectId,name:draft.name,root:draft.root,rows:draft.rows});draft=null;$('#annotation').hidden=true;}});
$('#roundForm').onsubmit=e=>{e.preventDefault();handle(async()=>{const f=new FormData(e.target),parameters={};$('#advanced').querySelectorAll('input').forEach(x=>{if(x.value!=='')parameters[x.name]=Number(x.value);});await act('create_round',{project:projectId,backbone:$('#backbone').value,seed:Number(f.get('seed')),tools:selectedEntryTools($('#roundForm'),!!project()?.active),train:chosen('#trainData'),val:chosen('#valData'),evaluation:chosen('#evalData'),initialization:f.get('initialization'),parameters});});};
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.start)handle(()=>act('start_round',{round:b.dataset.start}));if(b.dataset.cancel)handle(()=>act('cancel_round',{round:b.dataset.cancel,reason:tr('工程师取消本轮','Engineer cancelled round')}));if(b.dataset.log)handle(async()=>{await act('inspect',{project:projectId,purpose:'job_log',job:b.dataset.log});const r=await fetch('/api/log?job='+b.dataset.log);const v=await r.json();$('#jobLog').hidden=false;$('#jobLog').textContent=v.log||v.error;});if(b.dataset.interrupted)handle(()=>act('mark_interrupted',{job:b.dataset.interrupted,reason:$('#reason').value||'operator checked stopped worker PID'}));if(b.dataset.timer)handle(()=>act('timer',{project:projectId,command:b.dataset.timer}));});
$('#comparisonJob').onchange=renderComparison;
$('#decisionRound').onchange=renderCandidates;
async function decide(adopt){const j=state.jobs.find(x=>x.id===$('#comparisonJob').value),rid=$('#decisionRound').value;if(!rid||adopt&&!j)throw Error(tr('请选择轮次与候选','Select a round and candidate'));await act('decide',{round:rid,job:adopt?j.id:null,reason:$('#reason').value});}
$('#adopt').onclick=()=>handle(()=>decide(true));$('#hold').onclick=()=>handle(()=>decide(false));
$('#rollbackForm').onsubmit=e=>{e.preventDefault();handle(()=>act('rollback',{project:projectId,version:$('#rollbackVersion').value,reason:new FormData(e.target).get('reason'),expected_active:project()?.active}));};
$('#export').onclick=()=>handle(async()=>{await act('inspect',{project:projectId,purpose:'explicit_audit_export'});const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='facade-audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
handle(refresh);setInterval(()=>handle(refresh),5000);
