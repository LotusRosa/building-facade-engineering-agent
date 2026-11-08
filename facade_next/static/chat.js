// Conversational task cards use the same transaction/dedup path as the panels.
var conversationId=null,chatFresh=false,chatTargetProject=null,chatDraft=false,chatMessageSignature='',chatCardSignature='',chatDraftEdits=new Map(),chatPendingSelection=null,chatRequestPending=false;
function applyChatSelection(){if(!chatPendingSelection)return true;const target=chatPendingSelection,c=state.conversations?.find(x=>x.id===target.conversation);if(!c||c.messages.length<(target.message_count||0)||target.project&&!state.projects.some(p=>p.id===target.project))return false;if(typeof saveProjectForm==='function'&&projectId!==target.project)saveProjectForm();conversationId=target.conversation;projectId=target.project||'';chatFresh=false;chatTargetProject=null;chatPendingSelection=null;$('#chatSend').disabled=false;return true;}
function parkChatAnnotation(){if(chatDraft&&draft){collectAnnotations();chatDraftEdits.set(draft.id,{draft:structuredClone(draft),page});}}
function restorePanelAnnotation(){parkChatAnnotation();chatDraft=false;$('#intake').appendChild($('#annotation'));}
function resetChat(){parkChatAnnotation();conversationId=null;chatFresh=false;chatTargetProject=null;chatDraft=false;chatMessageSignature='';chatCardSignature='';}
function activeConversation(){if(chatFresh)return null;return state.conversations?.find(c=>c.id===conversationId)||state.conversations?.filter(c=>c.project===projectId).at(-1)||null;}
function roleName(role){return ({train:tr('Train · 训练/更新','Train · training/update'),val:tr('Val · 独立校准','Val · independent calibration'),evaluation:project()?.active?tr('本轮验收 · Current Gate','Acceptance · Current Gate'):tr('初始验收集 S0','Initial acceptance · S0')})[role];}
function chatHint(event){return ({create_project:tr('创建项目','Create project'),import_batch:tr('提交数据批次','Submit data batch'),begin_round:tr('开始新一轮维护','Start a new maintenance round'),import_folder:tr('导入照片文件夹','Import photo folder'),configure:tr('按所选配置准备任务','Prepare the selected task'),annotation_save:tr('保存本次标注','Save annotations'),propose_decision:tr('提交我的采用/保留选择','Submit my adoption/Hold choice'),confirm:tr('确认所列操作','Confirm the listed action'),reason:tr('补充决定理由','Provide the decision reason'),rollback:tr('准备版本回滚','Prepare version rollback')})[event]||event||'';}
function localQueryEvent(text){
 const key=text.trim().toLowerCase().replace(/[。.!！?？]+$/,'').replace(/^please\s+/,'').replace(/^请\s*/,'').replace(/\s+/g,' ');
 if(["what's the result","what is the result","what are the results",'compare','compare results' ,'show results','view results','results','view evaluation results','show evaluation results','view model comparison','show model comparison','查看结果','查看评估结果','查看评价结果','查看比较结果','比较结果','查看对比结果','查看模型对比'].includes(key))return 'compare';
 if(['status','show status','check status','progress','show progress','查看状态','查询状态','状态','查看进度','训练进度','查看训练进度'].includes(key))return 'status';
 return null;
}
function syncChatBusy(){
 const busy=chatRequestPending||!!chatPendingSelection,card=$('#chatCard');
 $('#chatSend').disabled=busy;$('#chatSend').textContent=busy?tr('处理中…','Processing…'):tr('发送','Send');
 card.setAttribute?.('aria-busy',String(busy));
 for(const b of card.querySelectorAll('button')){if(busy&&!b.disabled){b.dataset.pendingDisabled='1';b.disabled=true;}else if(!busy&&b.dataset.pendingDisabled){delete b.dataset.pendingDisabled;b.disabled=false;}}
}
async function sendChat(text,event=null,inputs={}){
 if(chatRequestPending){notice(()=>tr('上一条请求正在处理，完成后即可继续。','The previous request is processing; continue when it completes.'));return null;}
 if(chatPendingSelection){await refresh();if(chatPendingSelection)throw Error(tr('上一条已保存，等待最新状态恢复后再继续。','The previous message was saved; wait for fresh state before continuing.'));}
 if(chatDraft&&draft&&['按未知标签导入','保留未知标签导入','import with unknown labels'].includes(text.toLowerCase().trim())){collectAnnotations();event='annotation_save';inputs={draft:draft.id,rows:draft.rows};}
  if(!event)event=localQueryEvent(text);
 const c=activeConversation();
 chatRequestPending=true;syncChatBusy();notice(()=>tr('正在处理请求…','Processing request…'));
 try{
  const result=await act('chat',{conversation:c?.id||null,project:c?c.project:chatFresh?chatTargetProject:projectId||null,text,event,inputs,language:lang});
  chatPendingSelection=result;
  if(applyChatSelection()){chatCardSignature='';render();switchTab('conversation');}
  else notice(()=>tr('消息已保存；最新对话尚未刷新，恢复后再继续。','Message saved; waiting for the latest conversation before continuing.'));
  return result;
 }finally{chatRequestPending=false;syncChatBusy();}
}
function renderChat(){
 if(!state)return;
 const c=activeConversation(),p=c?state.projects.find(p=>p.id===c.project):chatFresh?state.projects.find(p=>p.id===chatTargetProject):project();if(c)conversationId=c.id;
 $('#chatOverview').innerHTML=`<strong>${esc(p?.name||tr('新项目','New project'))}</strong><dl><dt>${tr('实际在役','Actual active')}</dt><dd>${p?.active?esc(state.versions.find(v=>v.id===p.active)?.contract.backbone||p.active):tr('尚未建立','Not built yet')}</dd><dt>${tr('历史训练','Historical train')}</dt><dd>${p?.train_pool.length||0}</dd><dt>${tr('历史验收','Historical acceptance')}</dt><dd>${p?.eval_pool.length||0}</dd></dl>`;
 const signature=JSON.stringify([c?.id,c?.messages,lang]);
 if(signature!==chatMessageSignature){
  chatMessageSignature=signature;
   const messages=c?.messages||[{role:'assistant',reply:{zh:p?`继续 ${p.name}。告诉我你想做什么，也可以说“下一步”或“开始新一轮维护”。`:'你好，准备开始哪个项目？例如：“创建项目 东楼巡检”。',en:p?`Continue ${p.name}. Tell me what you need, or say “next” or “start a new maintenance round”.`:'What project shall we work on? Try “create project East building”.'}}];
  $('#chatMessages').innerHTML=messages.map(m=>`<article class="message ${m.role}"><span class="message-author">${m.role==='user'?tr('你','You'):'building-facade-engineering-agent'}</span><div>${esc(m.role==='user'?(m.text||chatHint(m.event)):m.reply?.[lang]||'').replace(/\n/g,'<br>')}</div></article>`).join('');
  $('#chatMessages').scrollTop=$('#chatMessages').scrollHeight;
 }
 let card=c?.card;if(!card)card={kind:!p?'create_project':'next'};
 const jobStates=card.kind==='status'?state.jobs.filter(j=>j.round===card.round).map(j=>[j.id,j.status,j.device,j.error,j.progress]):null;
 const key=JSON.stringify([c?.id,card,lang,state.datasets.filter(d=>d.project===projectId).map(d=>[d.id,d.name]),p?.active,jobStates]);
 if(key===chatCardSignature){syncChatBusy();return;}
 parkChatAnnotation();
 chatCardSignature=key;
 if(card.kind!=='annotation'&&chatDraft){$('#annotation').hidden=true;chatDraft=false;draft=null;}
 $('#chatCard').innerHTML=cardHTML(card);
 if(card.kind==='folder')$('#chatCard').insertAdjacentHTML('beforeend',`<div class="actions"><button type="button" class="secondary" data-chat-template="xlsx">${tr('下载 Excel 标注模板','Download Excel template')}</button><button type="button" class="secondary" data-chat-template="csv">${tr('下载 CSV 模板','Download CSV template')}</button></div>`);
 if(['folder','configure'].includes(card.kind)){$('#chatCard').innerHTML=`<details class="task-details"><summary>${card.kind==='folder'?tr('数据准备与导入','Prepare and import data'):tr('查看或调整本轮配置','Inspect this task configuration')}</summary>${$('#chatCard').innerHTML}</details>`;}
 if(card.kind==='annotation'){
  const saved=chatDraftEdits.get(card.draft.id);draft=saved?structuredClone(saved.draft):structuredClone(card.draft);chatDraft=true;page=saved?.page||0;$('#chatAnnotationSlot').appendChild($('#annotation'));$('#annotation').hidden=false;renderPhotos();
 }
 if(card.kind==='comparison')renderChatComparison(card);
 syncChatBusy();
 const scroll=$('.chat-scroll');if(scroll)scroll.scrollTop=scroll.scrollHeight;
}
function datasetOptions(selected=[]){return state.datasets.filter(d=>d.project===projectId).map(d=>`<option value="${d.id}" ${selected.includes(d.id)?'selected':''}>${esc(d.name)} (${d.count})</option>`).join('');}
function preparationGuide(){return `<details class="data-preparation"><summary>${tr('提交前：初始建模与后续轮次如何划分','Before submission: initial and later-round splits')}</summary><p>${tr('请由用户预先划分好数据，系统不自动划分或重排。','Prepare the splits yourself; the system does not split or reshuffle data.')}</p><ul><li>${tr('初始建模：Train（训练）＋Val（检查点选择/阈值校准）＋初始验收集 S0（独立评价）。','Initial: Train (learning) + Val (checkpoint selection/calibration) + initial acceptance S0 (independent evaluation).')}</li><li>${tr('后续A、B、C……每轮：新增Train＋本轮验收集Current Gate。已有Val、历史训练和历史验收自动沿用；新增独立Val可按需另交。','Each later round A, B, C…: new Train + independent Current Gate. Existing Val, training history and acceptance history are reused; additional independent Val is optional.')}</li><li>${tr('可在对话中提交整个已划分批次文件夹，核对子目录用途后分别导入；也可逐份导入子集。不需要额外final test。','Submit one pre-split batch folder in chat and confirm its subset mapping, or import subsets individually. No additional final-test dataset is needed.')}</li><li>${tr('按建筑/采集来源检查角色隔离，避免同图、重复图及相近连拍跨角色；未知标签不等于阴性。数据量和类别覆盖不足时应补充，比例不硬编码。','Keep buildings/source groups and duplicate/nearby captures separate across roles. Unknown labels are not negative. Supply enough data/class coverage; no split ratio is hard-coded.')}</li></ul></details>`;}
function cardHTML(card){
 const heading=t=>`<h3>${t}</h3>`;
  if(card.kind==='confirm_batch')return heading(tr('核对数据批次：','Review data batch: ')+esc(card.name))+`<p>${esc(card.summary[lang])}</p>`+table([tr('用途','Role'),tr('已有子目录','Existing subset'),tr('照片数','Photos')],card.parts.map(x=>[esc(x.role==='evaluation'?(card.stage==='initial'?'S0':'Current Gate'):x.role),esc(x.folder),x.count]))+`<button data-chat-confirm="${esc(card.token)}">${tr('确认分别导入子集','Confirm subset imports')}</button> <button class="secondary" data-chat-cancel-batch="${esc(card.token)}">${tr('取消本次导入','Cancel this intake')}</button>`;
  if(card.kind==='folder'&&card.batch)return heading(card.stage==='initial'?tr('提交初始数据批次','Submit the initial data batch'):tr('提交新一轮维护数据','Submit the next maintenance batch'))+preparationGuide()+`<p>${card.stage==='initial'?tr('一个批次文件夹内准备 train / val / safety（或 test）。名称可以是“首轮”“初始资料”等。','Prepare train / val / safety (or test) inside one batch folder. Its name is your choice.'):tr('一个新批次文件夹内准备 train / gate（或 test）；val可交，也可沿用已有Val。后续每轮都可以通过聊天继续。','Prepare train / gate (or test) inside one new batch folder. val is optional; existing Val is reused. Start each subsequent round in chat.')}</p><form data-chat-form="import_batch"><label>${tr('本机上的批次文件夹','Batch folder on the Agent host')}<input name="root" placeholder="${card.stage==='initial'?'/home/test/data/initial':'/home/test/data/round_A'}" required></label><label>${tr('批次名称（可选，默认使用文件夹名）','Batch name (optional, defaults to folder name)')}<input name="name"></label><button>${tr('识别已有划分并核对','Inspect existing subsets')}</button></form><details><summary>${tr('也可分别导入一个子集','Import individual subsets instead')}</summary>${cardHTML({...card,batch:false})}</details>`;
  if(card.kind==='create_project')return '';
 if(card.kind==='next')return '';
  if(card.kind==='folder')return heading(tr('导入已划分好的数据子集','Import a pre-split data subset'))+preparationGuide()+`<form data-chat-form="import_folder"><div class="grid"><label>${tr('这批数据用于','Data role')}<select name="role">${['train','val','evaluation'].map(r=>`<option value="${r}" ${r===card.role?'selected':''}>${roleName(r)}</option>`).join('')}</select></label><label>${tr('数据集/场景名称','Dataset / scene name')}<input name="name" value="${esc(roleName(card.role))}" required></label><label class="wide">${tr('对应子集文件夹（可含 images 子文件夹）','Subset folder (may contain an images subfolder)')}<input name="root" placeholder="/home/test/inspection_initial/train" required></label><label class="wide">${tr('标注文件（可留空，自动识别 labels.xlsx/csv）','Annotation file (optional; labels.xlsx/csv detected automatically)')}<input name="manifest"></label></div><button>${tr('检查并继续','Check and continue')}</button></form><small>${tr('完整标签跳过重复标注；缺失标签不会当作0。','Complete labels skip repeat annotation; missing labels never become0.')}</small>`;
 if(card.kind==='annotation')return heading(tr('在聊天里补齐标注','Complete annotations here'))+`<p>${tr('1=阳性，0=已确认阴性，空白=未知。可以同时存在多个缺陷。','1=positive,0=confirmed negative, blank=unknown. Multiple defects may coexist.')}</p><button class="secondary" data-chat-text="按未知标签导入" data-chat-en="import with unknown labels">${tr('明确保留未知标签并导入','Explicitly retain unknown labels and import')}</button>`;
 if(card.kind==='configure'){
  const selected=card.selected||{},opts=card.options||{},active=state.versions.find(v=>v.id===project()?.active),backbone=active?.contract.backbone||opts.backbone||'convnext_tiny',tools=inheritedEntryTools(opts.tools,!!active);
  return heading(tr('本次模型任务','This model task'))+`<form data-chat-form="configure"><div class="grid"><label>${tr('主干','Backbone')}<select name="backbone" ${active?'disabled':''}>${state.backbones.map(b=>`<option ${b===backbone?'selected':''}>${b}</option>`).join('')}</select></label><label>${tr('单个种子','Single seed')}<input type="number" name="seed" value="${opts.seed??20260921}" min="0" max="4294967295" required></label>${['train','val','evaluation'].map(r=>`<label>${roleName(r)}<select multiple name="${r}" ${r==='val'&&project()?.val_pool.length?'':'required'}>${datasetOptions(selected[r])}</select></label>`).join('')}<label>${tr('权重来源','Weight source')}<input type="hidden" name="initialization" value="cached_imagenet"><span>${active?tr('从当前在役版本继续更新','Continue from active version'):tr('ImageNet 预训练（本机缓存）','ImageNet pretrained (local cache)')}</span></label></div><fieldset><legend>${tr('训练入口','Training entries')}</legend><div class="tools">${state.tools.map(t=>`<label><input type="checkbox" name="tool" value="${t}" ${tools.includes(t)?'checked':''} ${!active||t==='core'?'disabled':''}>${esc(methodName(t))}</label>`).join('')}</div></fieldset><p>${active?tr('Standard 与其余 7 种方法用于维护，Core 仅用于首轮。','Standard and seven other methods are available for maintenance; Core is initial-only.'):tr('首轮固定使用 Core，从 ImageNet 预训练权重建立初始模型。','The initial round uses Core with ImageNet pretrained weights.')}</p><button>${tr('准备任务，先核对摘要','Prepare task and review summary')}</button></form>`;
 }
 if(card.kind.startsWith('confirm_'))return heading(tr('由你确认后执行','Confirm to execute'))+`<p>${esc(card.summary[lang])}</p><button data-chat-confirm="${card.token}">${tr('确认','Confirm')} · ${esc(({start:tr('启动候选任务','Start candidates'),adopt:tr('采用候选','Adopt candidate'),hold:tr('保留父模型','Hold parent'),cancel:tr('取消本轮','Cancel round'),rollback:tr('回滚版本','Rollback version')})[card.action])}</button>`;
 if(card.kind==='status'){
  const jobs=card.jobs.map(x=>({...x,...(state.jobs.find(j=>j.id===x.id)||{}),round:card.round})),initial=resultStage(card.round)==='initial';
  return heading(tr('实际任务状态','Actual job status'))+table([tr('方法','Method'),tr('状态','Status'),'GPU'],jobs.map(j=>[esc(methodName(j.method,j)),esc(j.status),esc(j.device||'—')]))+jobs.map(j=>jobProgressHTML(j)).join('')+(jobs.some(j=>j.status==='succeeded')?`<button type="button" data-chat-query="compare" data-round="${esc(card.round)}">${initial?tr('查看评估结果','View evaluation results'):tr('查看模型对比','View model comparison')}</button>`:'');
 }
 if(card.kind==='reason')return heading(tr('记录工程师理由','Record the engineer reason'))+`<form data-chat-form="reason" data-round="${esc(card.round)}"><textarea name="reason" rows="2" required placeholder="${tr('说明新旧场景与各类取舍','Describe the scene/class tradeoffs')}"></textarea><button>${tr('准备确认','Review decision')}</button></form>`;
 if(card.kind==='comparison'){
  const ready=card.jobs.some(j=>j.result?.status==='ready');
  const form=!card.closed&&(ready||card.can_hold)?`<form data-chat-form="propose_decision" data-round="${esc(card.round)}"><label>${tr('你的选择','Your choice')}<select id="chatDecisionChoice" name="choice">${ready?`<option value="adopt">${tr('采用所选候选','Adopt selected candidate')}</option>`:''}${card.can_hold?`<option value="hold">${tr('Hold，保留父模型','Hold the parent')}</option>`:''}</select></label><label>${tr('理由','Reason')}<textarea name="reason" rows="2" required></textarea></label><button>${tr('核对我的决定','Review my decision')}</button></form>`:'';
   const initial=resultStage(card.round)==='initial';
   return heading(resultsTitle(card.round))+(initial?`<p>${tr('初始轮查看 S0 验收指标，不进行父模型对比。核对后可决定是否采用。','Review the initial model’s S0 metrics. There is no parent comparison in this round; adoption follows your review.')}</p>`:'')+`<label>${initial?tr('查看模型','View model'):tr('查看候选','View candidate')}<select id="chatCandidate">${card.jobs.map(j=>`<option value="${j.id}">${esc(methodName(j.method,{...j,round:card.round}))}</option>`).join('')}</select></label><div id="chatComparison"></div>`+form;
 }
 if(card.kind==='rollback')return heading(tr('回滚到曾采用版本','Roll back to an adopted version'))+`<form data-chat-form="rollback"><select name="version" required>${card.versions.map(v=>`<option value="${v.id}">${esc(v.id+' · '+v.reason)}</option>`).join('')}</select><textarea name="reason" rows="2" required placeholder="${tr('回滚理由','Rollback reason')}"></textarea><button>${tr('核对回滚','Review rollback')}</button></form>`;
 return '';
}
function renderChatComparison(card){
 const id=$('#chatCandidate')?.value,j=card.jobs.find(x=>x.id===id),data=j?.result?.comparison,view=data?.candidate||data?.parent;
 const choice=$('#chatDecisionChoice');if(choice)choice.innerHTML=(j?.result?.status==='ready'?`<option value="adopt">${tr('采用所选候选','Adopt selected candidate')}</option>`:'')+(card.can_hold?`<option value="hold">${tr('Hold，保留父模型','Hold the parent')}</option>`:'');
 if(!view){$('#chatComparison').textContent=tr('当前没有已完成的评价结果。','No completed evaluation is available.');return;}
 const paired=!!data.candidate,groups=[['merged',view.merged,paired?data.parent?.merged:null],...Object.entries(view.scenes).map(([name,m])=>[name,m,paired?data.parent?.scenes[name]:null])];
 $('#chatComparison').innerHTML=(!paired?`<p>${tr('HNM 空池，没有新权重；以下为本轮父模型实际评价，可选择Hold。','HNM empty pool: no updated weights. These are actual parent evaluations; Hold is available.')}</p>`:'')+groups.map(([name,m,parent])=>`<h3>${esc(name)} · AP ${fmt(m.macro_AP)} (${m.AP_defined_classes}/3)</h3>`+table([tr('类别 / 版本','Class / version'),'AP','P','R','F1','FP','FN'],state.classes.flatMap(k=>{const row=(label,x)=>[esc(k)+' · '+label,...['AP','precision','recall','F1','FP','FN'].map(n=>fmt(x[n]))];return parent?[row(tr('父','parent'),parent.classes[k]),row(tr('候选','candidate'),m.classes[k])]:[row(paired?tr('候选','candidate'):tr('父','parent'),m.classes[k])];}))).join('');
}
async function saveChatAnnotation(){const currentDraft=draft;await sendChat(chatHint('annotation_save'),'annotation_save',{draft:currentDraft.id,rows:currentDraft.rows});}
$('#chatForm').onsubmit=e=>{e.preventDefault();const text=$('#chatInput').value.trim();if(!text)return;handle(async()=>{const key=typeof composerKey==='function'?composerKey():null;await sendChat(text);$('#chatInput').value='';if(key)composerDrafts.delete(key);});};
$('#newChatProject').onclick=()=>{if(chatDraft&&draft)collectAnnotations();conversationId=null;chatFresh=true;chatMessageSignature='';chatCardSignature='';renderChat();switchTab('conversation');};
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.chatQuery)handle(()=>sendChat(tr('\u67e5\u770b\u8bc4\u4f30\u7ed3\u679c','View evaluation results'),'compare',{round:b.dataset.round}));if(b.dataset.chatText)handle(()=>sendChat(lang==='zh'?b.dataset.chatText:b.dataset.chatEn));if(b.dataset.chatConfirm)handle(()=>sendChat(chatHint('confirm'),'confirm',{token:b.dataset.chatConfirm}));if(b.dataset.chatCancelBatch)handle(()=>sendChat(tr('取消本次导入','Cancel batch import'),'cancel_batch',{token:b.dataset.chatCancelBatch}));if(b.dataset.chatTemplate)handle(async()=>{await sendChat(tr('获取标注模板','Get annotation template'),'template',{format:b.dataset.chatTemplate});const a=document.createElement('a');a.href='/api/template?format='+b.dataset.chatTemplate;a.download='labels.'+b.dataset.chatTemplate;a.click();});});
document.addEventListener('submit',e=>{const kind=e.target.dataset.chatForm;if(!kind)return;e.preventDefault();const f=new FormData(e.target),inputs=Object.fromEntries(f);if(kind==='configure'){inputs.seed=Number(inputs.seed);inputs.backbone=e.target.querySelector('[name="backbone"]').value;try{inputs.tools=selectedEntryTools(e.target,!!project()?.active);}catch(error){notice(error.message,true);return;}for(const role of ['train','val','evaluation'])inputs[role]=f.getAll(role);delete inputs.tool;}if(kind==='propose_decision')inputs.job=$('#chatCandidate').value;if(kind==='propose_decision'||kind==='reason')inputs.round=e.target.dataset.round;handle(()=>sendChat(chatHint(kind),kind,inputs));});
document.addEventListener('change',e=>{if(e.target.id==='chatCandidate')renderChatComparison(activeConversation().card);});
