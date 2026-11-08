// Project/chat navigation. Workflow mutations still go through the shared service.
var workspaceView='conversation',projectFormDrafts=new Map(),composerDrafts=new Map(),projectFormBoundId=null,projectFormBoundActive=null;
function composerKey(){return conversationId&&!chatFresh?'chat:'+conversationId:'fresh:'+projectId;}
function parkComposer(){composerDrafts.set(composerKey(),$('#chatInput').value);}
function restoreComposer(){$('#chatInput').value=composerDrafts.get(composerKey())||'';}
function initializeWorkspaceForm(){const active=!!project()?.active;if(projectFormBoundId===projectId&&projectFormBoundActive===active)return;restoreProjectForm();projectFormBoundId=projectId;projectFormBoundActive=active;}
function projectChatContext(id){return (state.conversations||[]).find(c=>c.id===conversationId&&c.project===id)||(state.conversations||[]).filter(c=>c.project===id).at(-1)||null;}
function conversationTitle(c){const m=(c.messages||[]).find(m=>m.role==='user'&&m.text);return (m?.text||tr('新对话','New conversation')).slice(0,32);}
function projectParameterHTML(p){
 if(!p)return `<p class="empty-note">${tr('先创建或选择一个项目。','Create or select a project first.')}</p>`;
 const rs=(state.rounds||[]).filter(r=>r.project===p.id),r=[...rs].reverse().find(r=>!['decided','cancelled'].includes(r.status)),v=(state.versions||[]).find(v=>v.id===p.active),ctx=projectChatContext(p.id)?.context||{},opts=ctx.options||{},protocol=r?.protocol||(p.active?state.update_protocol:state.initial_protocol)||{};
 const count=role=>r?.roles?.[role]?.length??(ctx.selected?.[role]||[]).reduce((n,id)=>n+((state.datasets||[]).find(d=>d.id===id)?.count||0),0);
 const items=[
 [tr('建模阶段','Stage'),(r?.kind==='update'||!r&&p.active)?tr('持续维护','Maintenance'):tr('初始建模','Initial modeling')],
 [tr('主干模型','Backbone'),v?.contract?.backbone||r?.backbone||opts.backbone||'convnext_tiny'],
 [tr('权重来源','Weight source'),p.active?tr('当前在役版本 · ','Active version · ')+p.active:tr('ImageNet 预训练','ImageNet pretrained')],
  [tr('本轮训练入口','Training entries'),(r?.entry_tools||r?.tools||inheritedEntryTools(opts.tools,!!p.active)).map(t=>methodName(t,r||{kind:p.active?'update':'initial'})).join(', ')],
 ['Epochs',protocol.epochs??'—'],[tr('学习率','Learning rate'),protocol.lr??'—'],
 [tr('有效批次大小','Effective batch'),protocol.effective_batch??16],[tr('微批次大小','Micro batch'),protocol.micro_batch??8],
 [tr('本轮训练图片','Current training photos'),count('train')],[tr('本轮可用 Val','Current Val photos'),count('val')],[tr('本轮验收图片','Current acceptance photos'),count('evaluation')],
 [tr('历史训练图片','Historical training photos'),p.train_pool?.length||0],[tr('校准图片','Calibration photos'),p.val_pool?.length||0],
 [tr('历史验收图片','Historical acceptance photos'),p.eval_pool?.length||0],[tr('已有轮次','Recorded rounds'),rs.length]
 ];
 return `<div class="parameter-grid">${items.map(([label,value])=>`<div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('')}</div>`;
}
function renderWorkspace(){
 if(!state)return;
 const current=typeof activeConversation==='function'?activeConversation():null,selected=current?current.project:chatFresh?chatTargetProject:projectId;
 $('#projectList').innerHTML=state.projects.length?state.projects.map(p=>{
  const expanded=p.id===selected,cs=(state.conversations||[]).filter(c=>c.project===p.id);
  return `<div class="workspace-project ${expanded?'active':''}"><button class="project-row" data-workspace-project="${esc(p.id)}" aria-expanded="${expanded}"><span class="folder-icon" aria-hidden="true">▱</span><span>${esc(p.name)}</span><span class="project-chevron" aria-hidden="true">${expanded?'⌄':'›'}</span></button>${expanded?`<div class="project-children">${cs.map(c=>`<button class="conversation-row ${c.id===conversationId&&!chatFresh&&workspaceView==='conversation'?'selected':''}" data-workspace-conversation="${esc(c.id)}" data-project-id="${esc(p.id)}" title="${esc(conversationTitle(c))}">${esc(conversationTitle(c))}</button>`).join('')}<div class="project-links">${[['configure',tr('参数','Parameters')],['intake',tr('数据','Data')],['compare',resultsLabel(p.id)],['history',tr('\u5df2\u4fdd\u5b58\u4efb\u52a1','Saved tasks')],['audit',tr('版本与记录','Versions')]].map(([view,label])=>`<button data-workspace-view="${view}" class="${workspaceView===view?'selected':''}">${label}</button>`).join('')}</div></div>`:''}</div>`;
 }).join(''):`<p class="sidebar-empty">${tr('项目会显示在这里','Your projects appear here')}</p>`;
 const p=state.projects.find(p=>p.id===selected);
 const title=workspaceView==='settings'?tr('设置','Settings'):workspaceView==='conversation'?(current?conversationTitle(current):tr('新对话','New conversation')):({configure:tr('项目参数','Project parameters'),intake:tr('项目数据','Project data'),compare:resultsLabel(),audit:tr('版本与记录','Versions & audit')})[workspaceView];
 $('#workspaceProject').textContent=p?.name||'building-facade-engineering-agent';$('#workspaceTitle').textContent=title||tr('对话','Conversation');
 $('#projectParameters').innerHTML=projectParameterHTML(project());
}
function saveProjectForm(){
 if(!projectId)return;
 if(draft&&!chatDraft)collectAnnotations();
 projectFormDrafts.set(projectId,{backbone:$('#backbone').value,seed:$('#roundForm [name="seed"]').value,
  tools:$$('#tools input:checked').map(x=>x.value),advanced:$$('#advanced input').map(x=>[x.name,x.value]),
  data:['trainData','valData','evalData'].map(id=>[id,chosen('#'+id)]),
  annotation:chatDraft?(projectFormDrafts.get(projectId)?.annotation||null):draft?{draft:structuredClone(draft),page}:null});
}
function restoreProjectForm(){
 const saved=projectFormDrafts.get(projectId),p=project(),ctx=projectChatContext(projectId)?.context||{},r=[...(state.rounds||[])].reverse().find(r=>r.project===projectId&&!['decided','cancelled'].includes(r.status)),active=state.versions.find(v=>v.id===p?.active);
 $('#backbone').value=active?.contract.backbone||saved?.backbone||r?.backbone||ctx.options?.backbone||'convnext_tiny';
 $('#roundForm [name="seed"]').value=saved?.seed??r?.seed??ctx.options?.seed??20260921;
  const raw=saved?.tools??r?.entry_tools??r?.tools??ctx.options?.tools,tools=inheritedEntryTools(raw,!!active);
  $$('#tools input').forEach(x=>x.checked=tools.includes(x.value)&&(!x.disabled||x.value==='core'&&!active));
 $$('#advanced input').forEach(x=>x.value=saved?.advanced.find(([n])=>n===x.name)?.[1]||'');
 for(const id of ['trainData','valData','evalData']){const values=saved?.data.find(([name])=>name===id)?.[1]||r?.datasets?.[({trainData:'train',valData:'val',evalData:'evaluation'})[id]]||ctx.selected?.[({trainData:'train',valData:'val',evalData:'evaluation'})[id]]||[];$('#'+id).querySelectorAll('option').forEach(o=>o.selected=values.includes(o.value));}
 if(!chatDraft){draft=saved?.annotation?structuredClone(saved.annotation.draft):null;page=saved?.annotation?.page||0;$('#annotation').hidden=!draft;if(draft){$('#intake').appendChild($('#annotation'));renderPhotos();}}
}
function selectWorkspaceProject(id,chat=null){
 if(chatPendingSelection||chatRequestPending)throw Error(tr('上一条消息正在同步，请稍后切换。','The previous message is still synchronizing. Try switching shortly.'));
 if(!state.projects.some(p=>p.id===id))throw Error('Project no longer exists');
 if(chat&&!state.conversations.some(c=>c.id===chat&&c.project===id))throw Error('Conversation does not belong to this project');
  if(typeof AgentHistory!=='undefined')AgentHistory.close();
  parkComposer();saveProjectForm();
 resetChat();projectId=id;chatTargetProject=null;draft=null;$('#annotation').hidden=true;
 conversationId=chat||(state.conversations||[]).filter(c=>c.project===id).at(-1)?.id||null;
 workspaceView='conversation';render();restoreProjectForm();switchTab('conversation');renderWorkspace();restoreComposer();document.body.classList.remove('sidebar-open');
}
function startWorkspaceChat(){
 if(chatPendingSelection||chatRequestPending)throw Error(tr('上一条消息正在同步，请稍后新建。','The previous message is still synchronizing. Try again shortly.'));
 if(typeof AgentHistory!=='undefined')AgentHistory.close();
 parkComposer();saveProjectForm();resetChat();chatFresh=true;chatTargetProject=projectId||null;composerDrafts.delete(composerKey());draft=null;$('#annotation').hidden=true;
 workspaceView='conversation';render();restoreProjectForm();switchTab('conversation');renderWorkspace();restoreComposer();document.body.classList.remove('sidebar-open');
}
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;
 if(b.dataset.workspaceProject)handle(()=>selectWorkspaceProject(b.dataset.workspaceProject));
 if(b.dataset.workspaceConversation)handle(()=>selectWorkspaceProject(b.dataset.projectId,b.dataset.workspaceConversation));
 if(b.dataset.workspaceView==='history'){if(typeof AgentHistory!=='undefined')handle(()=>AgentHistory.open());return;}
 if(b.dataset.workspaceView){if(typeof AgentHistory!=='undefined')AgentHistory.close();workspaceView=b.dataset.workspaceView;switchTab(workspaceView);renderWorkspace();document.body.classList.remove('sidebar-open');}
});
$('#newChatProject').onclick=()=>handle(startWorkspaceChat);
$('#sidebarToggle').onclick=()=>document.body.classList.toggle('sidebar-open');
if(state)renderWorkspace();
