/* Folder intake and one project-specific label format. No image data is sent to an LLM. */
Object.assign(copy.zh, {
  folder_title:"图片文件夹与标签表格", folder_hint:"选择文件夹，或散选图片。支持 JPEG/PNG；子文件夹中的图片也会导入，文件名必须唯一。",
  choose_folder:"选择图片文件夹", optional_table:"标签表格（可选）", table_hint:"使用本项目模板：filename、每个类别的 0/1 列、no_defect。全部标签留空表示未标注；负样本必须填写 no_defect=1。",
  download_template:"下载 Excel 模板", export_labels:"导出 Excel 标注", export_csv:"导出 CSV 标注",
  table_preview:"预览并导入标签", legacy_zip:"兼容旧版 ZIP 输入", folder_confirm:"确认导入所选图片？图片只复制到本地项目目录，不会启动训练。",
  table_confirm:"确认应用这些标签？完整标注会自动执行数据校验并冻结通过校验的数据，跳过人工标注；不会启动训练。",
  labels_count:"将导入标签", overwrite_count:"将更新已标标签", remaining_count:"剩余未标注",
  duplicate_names:"文件名重复（包括子文件夹）。请先重命名，避免表格匹配错误：",
  table_ready:"标签已导入", table_failed:"图片已保留，但标签表格导入失败。请修正表格后重试，或直接在 Agent 内标注。",
  jump_unlabeled:"跳到下一张未标注", next_image:"下一张", shortcut_hint:"字母键切换标签；Enter 保存并继续。负样本与缺陷互斥。",
  all_labeled:"所有图片已标注。请校验后继续；不需要重复标注。", add_images:"继续添加图片", table_example:"统一表格示例", initial_split_hint:"首次建议 Train : 初始 Core Safety＝7:3 或 8:2，不强制。这里只导入 Train；Core Safety 单独预留，首次独立评估时导入。请保证类别覆盖，将同一建筑/连续采集的相关图片放在同组。",
  table_selected:"检测到标签表格", multiple_tables:"检测到多份标签表格，请选择本次使用的文件。", no_table:"未检测到标签表格，可在 Agent 内标注。", csv_template:"CSV 模板", no_images_selected:"请先选择 JPEG/PNG 图片。"
});
Object.assign(copy.en, {
  folder_title:"Image folder and annotation table", folder_hint:"Select a folder or individual images. JPEG/PNG images in subfolders are included; filenames must be unique.",
  choose_folder:"Select image folder", optional_table:"Annotation table (optional)", table_hint:"Use this project's template: filename, one 0/1 column per class, and no_defect. Blank labels mean unannotated; negative samples require no_defect=1.",
  download_template:"Download Excel template", export_labels:"Export Excel labels", export_csv:"Export CSV labels",
  table_preview:"Preview and import labels", legacy_zip:"Legacy ZIP compatibility", folder_confirm:"Import the selected images? Files are copied only into the local project. No training will start.",
  table_confirm:"Apply these labels? Complete labels trigger dataset checks and freeze the dataset if valid, skipping manual annotation. No training will start.",
  labels_count:"Labels to import", overwrite_count:"Existing labels to update", remaining_count:"Remaining unannotated",
  duplicate_names:"Repeated filenames (including subfolders). Rename them before import: ",
  table_ready:"Labels imported", table_failed:"Images are retained, but the annotation table failed. Correct the table and retry, or annotate directly inside the Agent.",
  jump_unlabeled:"Next unannotated image", next_image:"Next image", shortcut_hint:"Letter keys toggle labels; Enter saves and advances. No defect is exclusive with defects.",
  all_labeled:"All images are annotated. Validate to continue; repeat annotation is unnecessary.", add_images:"Add more images", table_example:"Unified table example",
  table_selected:"Annotation table detected", multiple_tables:"Multiple annotation tables detected. Select the table to use.", no_table:"No annotation table detected. Annotate inside the Agent.", csv_template:"CSV template", no_images_selected:"Select JPEG/PNG images first."
});

copy.en.initial_split_hint="Initially suggest Train : initial Core Safety = 7:3 or 8:2, not mandatory. Import only Train here; reserve Core Safety separately for the first paired evaluation. Ensure class coverage and keep related building/acquisition images in the same group.";

function tableDownload(format="xlsx", template=false, datasetId=app.dataset?.dataset_id) {
  const params=new URLSearchParams({project_id:app.project.project_id,format,template:String(template)});
  if(datasetId)params.set("dataset_id",datasetId);
  return `/api/annotations/table?${params}`;
}
function tableHeaderNames() {
  const names=app.project.classes.map(c=>c.display_name);
  return names.some(n=>["filename","no_defect"].includes(n.trim().toLowerCase()))?names.map(n=>`class:${n}`):names;
}
function tableHelp() {
  const headers=["filename",...tableHeaderNames(),"no_defect"],count=headers.length-2;
  const positive=["image_001.jpg",...Array.from({length:count},(_,i)=>i===0?1:0),0];
  const negative=["image_002.jpg",...Array(count).fill(0),1];
  return `<details class="table-example"><summary>${t("table_example")}</summary><div class="table-scroll"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${[positive,negative,["image_003.jpg",...Array(count+1).fill("")]].map(row=>`<tr>${row.map(v=>`<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>`;
}
function annotationTablePanel() {
  return `<section class="panel"><h4>${t("folder_title")}</h4><p>${t("table_hint")}</p><div class="form-actions"><a class="ghost-button" href="${tableDownload("xlsx",true)}">${t("download_template")}</a><a class="ghost-button" href="${tableDownload("xlsx")}">${t("export_labels")}</a><a class="ghost-button" href="${tableDownload("csv")}">${t("export_csv")}</a></div>${tableHelp()}<form id="label-table-form"><label class="field"><span>${t("optional_table")}</span><input id="label-table-file" type="file" accept=".xlsx,.csv" required></label><button class="primary-button" type="submit">${t("table_preview")}</button></form></section>`;
}
function folderImportPanel(target) {
  const maintenance=target==="maintenance",dataset=maintenance?app.activeBatch?.dataset:app.dataset;
  return hero(t(maintenance?"maintenance_import_title":"import_title"),t("folder_hint"))+(maintenance?"":`<p class="notice">${t("initial_split_hint")}</p>`)+`<form id="folder-import-form" class="panel">
    <label class="field"><span>${t("dataset_name")}</span><input id="folder-dataset-name" required value="${esc(dataset?.name||(maintenance?`${app.activeBatch.name}-images`:"initial-training-v1"))}" ${dataset?"disabled":""}></label>
    <label class="field"><span>${t("choose_folder")}</span><input id="folder-image-files" type="file" webkitdirectory directory multiple></label>
    <label class="field"><span>${t("choose_images")}</span><input id="individual-image-files" type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" multiple></label>
    <p id="folder-file-summary">${t("folder_hint")}</p>
    <label class="field"><span>${t("optional_table")}</span><input id="folder-label-file" type="file" accept=".xlsx,.csv"></label><p>${t("table_hint")}</p>
    <div class="form-actions"><a class="ghost-button" href="${tableDownload("xlsx",true,dataset?.dataset_id)}">${t("download_template")}</a><a class="ghost-button" href="${tableDownload("csv",true,dataset?.dataset_id)}">${t("csv_template")}</a></div>${tableHelp()}
    <div id="folder-upload-progress" hidden><div class="progress"><span style="width:0"></span></div><small></small></div>
    <div class="form-actions"><button class="primary-button" type="submit">${t("import_images")}</button></div></form>
    <details class="panel"><summary>${t("legacy_zip")}</summary>${labeledBundleForm(target)}</details>`;
}
function selectedFolderInput() {
  const selected=[...$("#folder-image-files").files,...$("#individual-image-files").files];
  const files=selected.filter(f=>/\.(jpe?g|png)$/i.test(f.name));
  const tables=selected.filter(f=>/\.(xlsx|csv)$/i.test(f.name));
  return {files,tables,table:$("#folder-label-file").files[0]||(tables.length===1?tables[0]:null)};
}
function bindFolderImport(target) {
  const update=()=>{const {files,tables,table}=selectedFolderInput();$("#folder-file-summary").textContent=`${files.length} ${t("images")} · ${table?`${t("table_selected")}: ${table.name}`:t(tables.length>1?"multiple_tables":"no_table")}`};
  ["#folder-image-files","#individual-image-files","#folder-label-file"].forEach(id=>$(id).onchange=update);
  $("#folder-import-form").onsubmit=event=>{
    event.preventDefault();const selection=selectedFolderInput();
    if(!selection.files.length){toast(t("no_images_selected"));return}
    if(selection.tables.length>1&&!selection.table){toast(t("multiple_tables"));return}
    const seen=new Set(),duplicate=selection.files.find(f=>{const name=f.name.toLowerCase();if(seen.has(name))return true;seen.add(name);return false});
    if(duplicate){toast(t("duplicate_names")+duplicate.name);return}
    const context={projectId:app.project.project_id,batchId:app.activeBatch?.batch_id,target,name:$("#folder-dataset-name").value.trim(),dataset:target==="maintenance"?app.activeBatch?.dataset:app.dataset};
    showConfirm(t("import_images"),`${t("folder_confirm")}\n${selection.files.length} ${t("images")}`,()=>uploadFolder(context,selection));
  };
  $(`#${target}-labeled-bundle-form`).onsubmit=event=>importLabeledBundle(event,target);
}
async function uploadFolder(context,{files,table}) {
  const form=$("#folder-import-form"),box=$("#folder-upload-progress");
  form.querySelectorAll("input,button").forEach(n=>n.disabled=true);box.hidden=false;
  let dataset=context.dataset;
  try {
    if(!dataset) {
      const maintenance=context.target==="maintenance";
      const data=await api("/api/tool/run",{method:"POST",body:JSON.stringify({tool:maintenance?"create_maintenance_dataset":"import_dataset",arguments:maintenance?{batch_id:context.batchId,name:context.name}:{project_id:context.projectId,name:context.name,role:"initial_training"},actor_type:"human",actor_id:"local_engineer",confirmed:false})});
      dataset=data.result.dataset;
    }
    for(let i=0;i<files.length;i++) {
      box.querySelector("small").textContent=`${t("uploading")} ${i+1}/${files.length}: ${files[i].name}`;
      await api(`/api/datasets/upload?${new URLSearchParams({dataset_id:dataset.dataset_id,filename:files[i].name})}`,{method:"POST",body:files[i]});
      box.querySelector("span").style.width=`${((i+1)/files.length)*100}%`;
    }
    app.imageIndex=0;
    await refresh(context.projectId);
    if(table)await previewAnnotationTable(dataset.dataset_id,table,context.projectId);
  } catch(error) {toast(error.message);await refresh(context.projectId)}
}
async function previewAnnotationTable(datasetId,file,projectId=app.project.project_id) {
  try {
    const params=new URLSearchParams({dataset_id:datasetId,filename:file.name});
    const preview=(await api(`/api/annotations/table/import?${params}`,{method:"POST",body:file})).result;
    const message=`${t("table_confirm")}\n${t("labels_count")}: ${preview.imported_labels}\n${t("overwrite_count")}: ${preview.overwrite_count}\n${t("remaining_count")}: ${preview.remaining_unlabeled}`;
    showConfirm(t("table_preview"),message,async()=>{
      try {
        params.set("confirmed","true");params.set("preview_token",preview.preview_token);
        const result=(await api(`/api/annotations/table/import?${params}`,{method:"POST",body:file})).result;
        app.lastReport=result.validation||null;toast(t("table_ready"));await refresh(projectId);
      } catch(error) {toast(error.message)}
    });
  } catch(error) {toast(`${t("table_failed")} ${error.message}`)}
}

// Existing renderWorkspace and maintenance routing use these shared entry points.
importPanel=()=>folderImportPanel("initial");
maintenanceImportPanel=()=>folderImportPanel("maintenance");
bindImport=()=>bindFolderImport("initial");
bindMaintenanceImport=()=>bindFolderImport("maintenance");
const renderAnnotationWithoutTables=renderAnnotation;
renderAnnotation=function() {
  if(app.annotationDataset!==app.dataset?.dataset_id){app.annotationDataset=app.dataset?.dataset_id;app.imageIndex=0}
  renderAnnotationWithoutTables();
  if(!$("#annotation-form"))return;
  const form=$("#annotation-form");
  form.closest(".annotation-layout").insertAdjacentHTML("beforebegin",annotationTablePanel());
  $("#label-table-form").onsubmit=event=>{event.preventDefault();previewAnnotationTable(app.dataset.dataset_id,$("#label-table-file").files[0])};
  const inputs=[...form.querySelectorAll("input[type=checkbox]")];
  inputs.forEach((input,index)=>{if(index<26){input.dataset.shortcut=String.fromCharCode(65+index);input.nextElementSibling.insertAdjacentHTML("afterbegin",`<kbd>${input.dataset.shortcut}</kbd> `)}});
  form.insertAdjacentHTML("beforeend",`<p>${t("shortcut_hint")}</p><div class="form-actions"><button id="next-image" class="ghost-button" type="button">${t("next_image")}</button><button id="jump-unlabeled" class="ghost-button" type="button">${t("jump_unlabeled")}</button></div>`);
  $("#next-image").disabled=app.imageIndex>=app.images.length-1;
  $("#next-image").onclick=()=>{app.imageIndex++;renderWorkspace()};
  const pending=app.images.filter(i=>i.annotation_status!=="complete").length;
  $("#jump-unlabeled").disabled=!pending;
  $("#jump-unlabeled").onclick=()=>{for(let offset=1;offset<=app.images.length;offset++){const index=(app.imageIndex+offset)%app.images.length;if(app.images[index].annotation_status!=="complete"){app.imageIndex=index;renderWorkspace();break}}};
  if(!pending)form.insertAdjacentHTML("beforeend",`<div class="notice success">${t("all_labeled")}</div>`);
  $("#workspace-content").insertAdjacentHTML("beforeend",`<details class="panel"><summary>${t("add_images")}</summary>${folderImportPanel(app.dataset.role==="maintenance"?"maintenance":"initial")}</details>`);
  bindFolderImport(app.dataset.role==="maintenance"?"maintenance":"initial");
};
let annotationSaving=false;
const saveAnnotationWithoutGuard=saveAnnotation;
saveAnnotation=async function(event) {
  event.preventDefault();if(annotationSaving)return;annotationSaving=true;
  const button=$("#annotation-form button[type=submit]");if(button)button.disabled=true;
  try {await saveAnnotationWithoutGuard(event)}finally{annotationSaving=false;if(button)button.disabled=false}
};
document.addEventListener("keydown",event=>{
  if(event.ctrlKey||event.metaKey||event.altKey||event.repeat||document.querySelector("dialog[open]")||!$("#advanced-workspace").classList.contains("open"))return;
  if(event.target?.closest("textarea,select,[contenteditable=true],input:not([type=checkbox])"))return;
  const form=$("#annotation-form")||$("#evaluation-annotation-form");if(!form||annotationSaving)return;
  if(event.key==="Enter"){event.preventDefault();form.requestSubmit();return}
  const input=[...form.querySelectorAll("input[data-shortcut]")].find(n=>n.dataset.shortcut===event.key.toUpperCase());
  if(input){event.preventDefault();input.checked=!input.checked;input.dispatchEvent(new Event("change",{bubbles:true}))}
});
