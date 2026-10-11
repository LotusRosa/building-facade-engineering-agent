Object.assign(copy.zh,{evaluation_folder:"独立评估图片文件夹",evaluation_folder_hint:"首次评估导入预留的初始 Core Safety 和本轮 Current Gate；后续只需导入新的 Current Gate。累计 Core Safety 自动附加，Train 每轮另行准备。各用途图片不能重叠。",safety_seed:"初始 Core Safety（仅首次）",safety_history:"已有累计 Core Safety（自动附加）",evaluation_labels_hint:"请导入表格，或在下方补齐标注。全部标签完成后，确认冻结独立评估任务；不会自动启动 GPU 或切换模型。",cohort_label:"评估数据用途",freeze_folder_evaluation:"校验并冻结评估任务"});
Object.assign(copy.en,{evaluation_folder:"Independent evaluation image folders",evaluation_folder_hint:"For the first evaluation import the reserved initial Core Safety and fresh Current Gate. Later rounds supply only a fresh Current Gate; cumulative Core Safety is attached automatically. Prepare separate Train data every round. Images must not overlap across roles.",safety_seed:"Initial Core Safety (once only)",safety_history:"Existing cumulative Core Safety (attached automatically)",evaluation_labels_hint:"Import a table or complete labels below. After annotation, confirm the independent evaluation job. No GPU execution or model switch starts automatically.",cohort_label:"Evaluation cohort",freeze_folder_evaluation:"Validate and freeze evaluation job"});
function evaluationCohortTitle(cohort){return t(cohort==="current_gate"?"current_gate":"safety_seed")}
Object.assign(copy.zh,{evaluation_intro:"每轮同时评估本轮 Current Gate 与累计 Core Safety；初始 Core Safety 只提供一次，后续自动累计已完成轮次的 Gate。Train 始终单独准备且不与评估数据重叠。"});
Object.assign(copy.en,{evaluation_intro:"Every round evaluates Current Gate together with cumulative Core Safety. Supply the initial safety seed once; completed Gates accumulate automatically. Prepare separate Train data every round, with no evaluation overlap."});
function evaluationInputTableUrl(batchId,cohort,template=false){return `/api/evaluation/input/table?${new URLSearchParams({batch_id:batchId,cohort,format:"xlsx",template:String(template)})}`}
Object.assign(copy.zh,{round_data_title:"本轮数据准备：Train 与 Current Gate",round_data_hint:"已有 Champion，无需重新初始化。每轮准备互不重叠的维护/更新（Train）和 Current Gate 文件夹。初始 Core Safety 仅预留一次，首次评估时导入，之后由 Agent 自动累计已完成轮次的 Gate。每组可带一份 Excel/CSV，无标签可在 Agent 内标注。不强制固定比例，请保证类别覆盖和足够样本；评估数据绝不参与训练。",round_maintenance:"维护/更新 Train（筛查与训练）",round_evaluation:"独立评估数据（不训练）",evaluation_early_hint:"这些评估图片和标签会保留到本轮评估。Challenger 注册后才能冻结评估任务；不会提前运行评估。"});
Object.assign(copy.en,{round_data_title:"Prepare this round: Train and Current Gate",round_data_hint:"A Champion already exists; do not reinitialize it. Every round needs disjoint maintenance/update (Train) and Current Gate folders. Reserve initial Core Safety once and import it for the first evaluation; the Agent then accumulates completed Gates automatically. Each folder may include an Excel/CSV table; otherwise annotate inside the Agent. No fixed ratio is enforced: ensure class coverage and sufficient samples. Evaluation images never enter training.",round_maintenance:"Maintenance/update Train (screening and training)",round_evaluation:"Independent evaluation (no training)",evaluation_early_hint:"Evaluation images and labels are retained for this round. Freeze the evaluation job only after Challenger registration; evaluation does not run early."});
function evaluationFolderPanel(snapshot) {
  // Ratios guide engineers; they never trigger automatic data partitioning.
  if(!Object.hasOwn(snapshot.cohorts,app.evaluationInputCohort||"current_gate"))app.evaluationInputCohort="current_gate";
  const cohort=app.evaluationInputCohort||"current_gate",images=snapshot.images.filter(i=>i.cohort===cohort);
  const image=images[Math.min(app.evaluationInputIndex||0,Math.max(0,images.length-1))];
  return `<h4>${t("evaluation_folder")}</h4><p>${t("evaluation_folder_hint")}</p><div class="stat-grid">${Object.entries(snapshot.cohorts).map(([c,v])=>`<div class="stat"><strong>${v.labeled_count} / ${v.image_count}</strong><small>${evaluationCohortTitle(c)} · ${t("labeled")}</small></div>`).join("")}</div>
    <p>${t("safety_history")}: ${snapshot.core_safety_history_count||0} ${t("images")}</p><label class="field"><span>${t("cohort_label")}</span><select id="evaluation-input-cohort">${Object.keys(snapshot.cohorts).map(c=>`<option value="${c}" ${cohort===c?"selected":""}>${evaluationCohortTitle(c)}</option>`).join("")}</select></label>
    <form id="evaluation-folder-form"><label class="field"><span>${t("choose_folder")}</span><input id="evaluation-folder-files" type="file" webkitdirectory directory multiple></label><label class="field"><span>${t("choose_images")}</span><input id="evaluation-individual-files" type="file" multiple accept=".jpg,.jpeg,.png"></label><p id="evaluation-folder-summary">${t("folder_hint")}</p><label class="field"><span>${t("optional_table")}</span><input id="evaluation-folder-table" type="file" accept=".xlsx,.csv"></label><button class="primary-button" type="submit">${t("import_images")}</button><p id="evaluation-upload-progress" role="status"></p></form>
    <p>${t("table_hint")}</p>${tableHelp()}<div class="form-actions"><a class="ghost-button" href="${evaluationInputTableUrl(snapshot.batch_id,cohort,true)}">${t("download_template")}</a><a class="ghost-button" href="${evaluationInputTableUrl(snapshot.batch_id,cohort)}">${t("export_labels")}</a></div>
    <form id="evaluation-table-form"><label class="field"><span>${t("optional_table")}</span><input id="evaluation-table-file" type="file" accept=".xlsx,.csv" required></label><button class="ghost-button" type="submit">${t("table_preview")}</button></form>
    ${image?`<section class="annotation-layout"><div><div class="image-stage"><img src="/api/evaluation/input/image?image_id=${encodeURIComponent(image.image_id)}" alt="${esc(image.filename)}"></div><div class="image-meta">${esc(image.filename)}</div></div><form id="evaluation-annotation-form" class="panel"><div class="label-list">${app.project.classes.map((c,index)=>`<label class="label-option"><input class="evaluation-defect-check" type="checkbox" value="${esc(c.class_id)}" ${index<26?`data-shortcut="${String.fromCharCode(65+index)}"`:""} ${image.class_ids.includes(c.class_id)?"checked":""}><span>${index<26?`<kbd>${String.fromCharCode(65+index)}</kbd> `:""}${esc(c.display_name)}</span></label>`).join("")}<label class="label-option no-defect"><input id="evaluation-no-defect" type="checkbox" ${app.project.classes.length<26?`data-shortcut="${String.fromCharCode(65+app.project.classes.length)}"`:""} ${image.no_defect?"checked":""}><span>${app.project.classes.length<26?`<kbd>${String.fromCharCode(65+app.project.classes.length)}</kbd> `:""}${t("no_defect")}</span></label></div><div class="annotation-nav"><button id="evaluation-prev" class="ghost-button" type="button">${t("previous")}</button><span>${images.indexOf(image)+1} / ${images.length}</span><button class="primary-button" type="submit">${t("save_next")}</button></div><div class="form-actions"><button id="evaluation-next" class="ghost-button" type="button">${t("next_image")}</button><button id="evaluation-unlabeled" class="ghost-button" type="button">${t("jump_unlabeled")}</button></div><p>${t("shortcut_hint")}</p></form></section>`:""}
    <p>${t(app.activeBatch?.state==="CHALLENGER_TRAINED"?"evaluation_labels_hint":"evaluation_early_hint")}</p><button id="freeze-folder-evaluation" class="primary-button" type="button" ${snapshot.complete&&app.activeBatch?.state==="CHALLENGER_TRAINED"?"":"disabled"}>${t("freeze_folder_evaluation")}</button>`;
}
function evaluationSelectedFiles() {
  const selected=[...$("#evaluation-folder-files").files,...$("#evaluation-individual-files").files];
  const tables=selected.filter(f=>/\.(xlsx|csv)$/i.test(f.name));
  return {files:selected.filter(f=>/\.(jpe?g|png)$/i.test(f.name)),tables,table:$("#evaluation-folder-table").files[0]||(tables.length===1?tables[0]:null)};
}
async function loadEvaluationInput(batchId) {
  try {
    const snapshot=(await api(`/api/evaluation/input/status?batch_id=${encodeURIComponent(batchId)}`)).result;
    if(app.activeBatch?.batch_id!==batchId||app.evaluationJobs.length||!$("#evaluation-folder-panel"))return;
    if(app.evaluationInputBatch!==batchId){app.evaluationInputBatch=batchId;app.evaluationInputCohort="current_gate";app.evaluationInputIndex=0}
    $("#evaluation-folder-panel").innerHTML=evaluationFolderPanel(snapshot);
    bindEvaluationInput(snapshot);
  }catch(error){toast(error.message)}
}
function bindEvaluationInput(snapshot) {
  const cohort=app.evaluationInputCohort||"current_gate",batchId=snapshot.batch_id,projectId=snapshot.project_id;
  $("#evaluation-input-cohort").onchange=event=>{app.evaluationInputCohort=event.target.value;app.evaluationInputIndex=0;loadEvaluationInput(batchId)};
  const update=()=>{const {files,tables,table}=evaluationSelectedFiles();$("#evaluation-folder-summary").textContent=`${files.length} ${t("images")} · ${table?`${t("table_selected")}: ${table.name}`:t(tables.length>1?"multiple_tables":"no_table")}`};
  ["#evaluation-folder-files","#evaluation-individual-files","#evaluation-folder-table"].forEach(id=>$(id).onchange=update);
  $("#evaluation-folder-form").onsubmit=event=>{
    event.preventDefault();const selection=evaluationSelectedFiles();
    if(!selection.files.length){toast(t("no_images_selected"));return}
    if(selection.tables.length>1&&!selection.table){toast(t("multiple_tables"));return}
    const seen=new Set(),duplicate=selection.files.find(f=>{const key=f.name.toLowerCase();if(seen.has(key))return true;seen.add(key);return false});
    if(duplicate){toast(t("duplicate_names")+duplicate.name);return}
    showConfirm(t("import_images"),`${evaluationCohortTitle(cohort)} · ${selection.files.length} ${t("images")}\n${t("folder_confirm")}`,async()=>{
      const form=$("#evaluation-folder-form");form.querySelectorAll("input,button").forEach(n=>n.disabled=true);
      $("#evaluation-input-cohort").disabled=true;
      try {
        for(let i=0;i<selection.files.length;i++) {
          $("#evaluation-upload-progress").textContent=`${t("uploading")} ${i+1}/${selection.files.length}: ${selection.files[i].name}`;
          await api(`/api/evaluation/input/upload?${new URLSearchParams({batch_id:batchId,cohort,filename:selection.files[i].name,confirmed:"true"})}`,{method:"POST",body:selection.files[i]});
        }
        await loadEvaluationInput(batchId);
        if(selection.table)await previewEvaluationTable(batchId,cohort,selection.table);
      }catch(error){toast(error.message);await loadEvaluationInput(batchId)}
    });
  };
  $("#evaluation-table-form").onsubmit=event=>{event.preventDefault();previewEvaluationTable(batchId,cohort,$("#evaluation-table-file").files[0])};
  const images=snapshot.images.filter(i=>i.cohort===cohort),index=Math.min(app.evaluationInputIndex||0,Math.max(0,images.length-1)),image=images[index];
  if(image) {
    const noDefect=$("#evaluation-no-defect"),checks=[...document.querySelectorAll(".evaluation-defect-check")];
    noDefect.onchange=()=>{if(noDefect.checked)checks.forEach(c=>c.checked=false)};
    checks.forEach(c=>c.onchange=()=>{if(c.checked)noDefect.checked=false});
    $("#evaluation-prev").disabled=index===0;$("#evaluation-next").disabled=index===images.length-1;
    $("#evaluation-prev").onclick=()=>{app.evaluationInputIndex=index-1;loadEvaluationInput(batchId)};
    $("#evaluation-next").onclick=()=>{app.evaluationInputIndex=index+1;loadEvaluationInput(batchId)};
    $("#evaluation-unlabeled").disabled=images.every(i=>i.annotation_status==="complete");
    $("#evaluation-unlabeled").onclick=()=>{for(let offset=1;offset<=images.length;offset++){const next=(index+offset)%images.length;if(images[next].annotation_status!=="complete"){app.evaluationInputIndex=next;loadEvaluationInput(batchId);break}}};
    $("#evaluation-annotation-form").onsubmit=async event=>{
      event.preventDefault();if(annotationSaving)return;
      const classIds=checks.filter(c=>c.checked).map(c=>c.value),negative=noDefect.checked;
      if(!classIds.length&&!negative){toast(t("select_one"));return}
      annotationSaving=true;const button=event.currentTarget.querySelector("button[type=submit]");button.disabled=true;
      try{await api("/api/evaluation/input/save",{method:"POST",body:JSON.stringify({image_id:image.image_id,class_ids:classIds,no_defect:negative})});app.evaluationInputIndex=Math.min(index+1,images.length-1);await loadEvaluationInput(batchId)}catch(error){toast(error.message)}finally{annotationSaving=false;button.disabled=false}
    };
  }
  $("#freeze-folder-evaluation").onclick=()=>showConfirm(t("freeze_folder_evaluation"),t("evaluation_labels_hint"),async()=>{
    const button=$("#freeze-folder-evaluation");button.disabled=true;
    try {await api("/api/evaluation/input/freeze",{method:"POST",body:JSON.stringify({batch_id:batchId,confirmed:true,preview_token:snapshot.preview_token})});await refresh(projectId)}catch(error){toast(error.message);button.disabled=false}
  });
}
async function previewEvaluationTable(batchId,cohort,file) {
  try {
    const params=new URLSearchParams({batch_id:batchId,cohort,filename:file.name});
    const preview=(await api(`/api/evaluation/input/table/import?${params}`,{method:"POST",body:file})).result;
    showConfirm(t("table_preview"),`${t("labels_count")}: ${preview.imported_labels}\n${t("overwrite_count")}: ${preview.overwrite_count}\n${t("remaining_count")}: ${preview.remaining_unlabeled}\n${t("evaluation_labels_hint")}`,async()=>{
      try {params.set("confirmed","true");params.set("preview_token",preview.preview_token);await api(`/api/evaluation/input/table/import?${params}`,{method:"POST",body:file});await loadEvaluationInput(batchId);toast(t("table_ready"))}catch(error){toast(error.message)}
    });
  }catch(error){toast(`${t("table_failed")} ${error.message}`)}
}
const renderChallengerWithLegacyInput=renderChallengerTrained;
renderChallengerTrained=function(model) {
  renderChallengerWithLegacyInput(model);
  if(app.evaluationJobs.length||app.evidence||app.activeBatch?.state!=="CHALLENGER_TRAINED")return;
  const target=$("#workspace-content"),legacy=target.lastElementChild;
  const details=document.createElement("details");details.className="panel";
  const summary=document.createElement("summary");summary.textContent=t("legacy_zip");details.append(summary,legacy);target.append(details);
  details.insertAdjacentHTML("beforebegin",`<section id="evaluation-folder-panel" class="panel"><p>${t("evaluation_folder")}</p></section>`);
  loadEvaluationInput(app.activeBatch.batch_id);
};

const renderWorkspaceWithFolderInput=renderWorkspace;
renderWorkspace=function() {
  renderWorkspaceWithFolderInput();
  const batch=app.activeBatch;
  if(app.view!=="task"||!batch||app.evaluationJobs.length||!["CREATED","MAINTENANCE_DATA_IMPORTED","MAINTENANCE_LABELS_READY","MAINTENANCE_BATCH_FROZEN","FAILURE_DISCOVERY_COMPLETED","FAILURE_REVIEW_COMPLETED"].includes(batch.state))return;
  if(app.roundInputBatch!==batch.batch_id){app.roundInputBatch=batch.batch_id;app.roundInputView="maintenance"}
  const guide=`<section class="panel"><h4>${t("round_data_title")}</h4><p>${t("round_data_hint")}</p><div class="form-actions"><button id="round-maintenance-tab" class="${app.roundInputView==="maintenance"?"primary-button":"ghost-button"}" type="button">${t("round_maintenance")}</button><button id="round-evaluation-tab" class="${app.roundInputView==="evaluation"?"primary-button":"ghost-button"}" type="button">${t("round_evaluation")}</button></div></section>`;
  const target=$("#workspace-content");
  if(app.roundInputView==="evaluation") {
    target.innerHTML=guide+`<section id="evaluation-folder-panel" class="panel"><p>${t("evaluation_folder")}</p></section>`;
    loadEvaluationInput(batch.batch_id);
  }else target.insertAdjacentHTML("afterbegin",guide);
  $("#round-maintenance-tab").onclick=()=>{app.roundInputView="maintenance";renderWorkspace()};
  $("#round-evaluation-tab").onclick=()=>{app.roundInputView="evaluation";renderWorkspace()};
};

copy.zh.round_data_hint+="建议本轮 Train : Current Gate＝7:3 或 8:2，不强制；同一建筑或连续采集的相近图片应放在同组。";
copy.en.round_data_hint+=" Suggested Train : Current Gate = 7:3 or 8:2, not mandatory; keep related building or consecutive acquisition images together.";
