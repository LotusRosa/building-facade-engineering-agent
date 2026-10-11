/* Translate product-owned copy, never user labels, file names, IDs or worker logs. */
Object.assign(copy.en, {
  release_status:"AGENT 2.2 · COMPLETE",close:"Close",close_workflow:"Close workflow settings",close_task_panel:"Close task panel",
  compatible_connection:"OpenAI-compatible",base_url:"Base URL",api_key:"API Key",model_example:"e.g. your-model-name",
  api_key_example:"sk-... (optional for local model)",chat_protocol:"Chat Completions (local-compatible)",prepare_runtime:"Prepare runtime",
  missing_training_job:"Training job record is missing.",provider_custom:"Custom OpenAI-compatible",provider_local:"local",
  status_label:"Status",safety_seed:"Initial Core Safety (once only)",missing_model_record:"Model registry record is missing."
});
Object.assign(copy.zh, {
  assistant:"建筑立面工程智能助手",release_status:"智能助手 2.2 · 完整版",close:"关闭",close_workflow:"关闭项目流程",close_task_panel:"关闭任务面板",
  compatible_connection:"兼容 OpenAI 接口",base_url:"接口基础地址",api_key:"API 密钥",model_example:"例如：你的模型名称",
  api_key_example:"sk-...（本地模型可不填写）",chat_protocol:"对话补全接口（兼容本地模型）",prepare_runtime:"准备运行环境",
  missing_training_job:"缺少训练任务记录。",provider_custom:"自定义兼容 OpenAI 接口",provider_local:"本地",
  status_label:"状态",current_gate:"本轮评估集",new_holdout:"本轮评估集",historical_holdout:"累计安全评估集",
  historical_gates:"累计安全评估集",safety_seed:"初始安全评估集（仅首次）",safety_history:"已有累计安全评估集（自动附加）",
  gate_history:"安全评估集累计历史",no_defect:"无缺陷",macro_map:"宏平均 mAP",macro_f1:"宏平均 F1",micro_f1:"微平均 F1",
  false_positives:"误报",false_negatives:"漏报",deploy:"启用候选模型",hold:"保留当前模型",epoch_progress:"训练轮次",
  phase_next:"请按当前工作流继续。",missing_model_record:"缺少模型注册记录。"
});
const chineseTerms=[
  [/Initial Champion/g,"初始模型"],[/Active Champion/g,"当前模型"],[/Champion–Challenger/g,"当前模型与候选模型"],
  [/Current Gate/g,"本轮评估集"],[/Core Safety/g,"安全评估集"],[/Failure Slices?/g,"失败切片"],
  [/Challenger/g,"候选模型"],[/Champion/g,"当前模型"],[/remaining-new/gi,"其余新数据"],[/history replay/gi,"历史回放"],
  [/History replay/g,"历史回放"],[/Remaining-new/g,"其余新数据"],[/Train/g,"训练集"],[/Gate/g,"评估集"],
  [/Accept/g,"接受"],[/Trim/g,"剔除"],[/Reject/g,"拒绝"],[/Promote/g,"启用"],[/Retain/g,"保留"],[/Hold/g,"保留当前模型"],
  [/No defect/g,"无缺陷"],[/Worker/g,"运行进程"],[/Epoch/g,"训练轮次"],[/preset/g,"预设配置"],[/manifest/g,"清单"],
  [/failure/g,"失败记录"],[/Agent/g,"智能助手"]
];
for(const key of Object.keys(copy.zh))for(const [pattern,value] of chineseTerms)copy.zh[key]=copy.zh[key].replace(pattern,value);
Object.assign(copy.zh,{
  evaluation_title:"当前模型与候选模型独立评估",
  evaluation_intro:"每轮同时评估本轮评估集与累计安全评估集。初始安全评估集只提供一次；后续自动累计已完成轮次的评估集。训练集每轮单独准备，不能与评估数据重叠。",
  evaluation_folder_hint:"首次评估需导入预留的初始安全评估集和本轮评估集；后续只需导入新的本轮评估集，累计安全评估集自动附加。每轮训练集另行准备，各用途图片不能重叠。"
});
workflowKeys.zh=["创建项目","确认类别","导入与标注","数据检查","训练初始模型","维护数据","失败切片审核","保留或启用模型"];
function providerLabel(item){return item.id==="custom"?t("provider_custom"):item.label.replace("(local)",`(${t("provider_local")})`)}
const statusLabels={EMPTY:["尚无项目","No project"],PROJECT_CREATED:["项目已创建","Project created"],TAXONOMY_CONFIRMED:["类别已确认","Classes confirmed"],DATA_IMPORTED:["数据已导入","Data imported"],DATA_VALIDATED:["数据已校验","Data validated"],INITIAL_TRAINING_READY:["初始训练已准备","Initial training ready"],CHAMPION_READY:["初始模型已注册","Champion registered"],SCREENING_READY:["可开始维护","Ready for maintenance"],MAINTENANCE_BATCH_CREATED:["维护批次已创建","Maintenance batch created"],MAINTENANCE_DATA_IMPORTED:["维护数据已导入","Maintenance data imported"],MAINTENANCE_BATCH_FROZEN:["维护数据已冻结","Maintenance data frozen"],FAILURE_DISCOVERY_COMPLETED:["失败发现已完成","Failure discovery completed"],FAILURE_REVIEW_COMPLETED:["专家审核已完成","Expert review completed"],CHALLENGER_JOB_READY:["候选训练已准备","Challenger job ready"],CHALLENGER_TRAINED:["候选模型已注册","Challenger registered"],EVIDENCE_READY:["评估证据已准备","Evaluation evidence ready"],DEPLOYED:["候选模型已启用","Challenger promoted"],HELD:["当前模型已保留","Champion retained"],queued:["排队中","Queued"],starting:["正在启动","Starting"],running:["运行中","Running"],completed:["已完成","Completed"],result_verified:["结果已校验","Result verified"],failed:["失败","Failed"],cancelled:["已取消","Cancelled"],interrupted:["已中断","Interrupted"],pending:["待处理","Pending"],approved:["已确认","Approved"],rejected:["已拒绝","Rejected"],frozen:["已冻结","Frozen"],open:["待准备","Open"],validated:["已校验","Validated"],ready:["已准备","Ready"],created:["已创建","Created"],active:["已生效","Active"],retired:["已停用","Retired"],promote:["启用候选模型","Promote Challenger"],retain:["保留当前模型","Retain Champion"],deploy:["启用候选模型","Promote Challenger"],hold:["保留当前模型","Retain Champion"]};
function localizedStatus(value){return statusLabels[value]?.[app.lang==="zh"?0:1]||value||"—"}
refresh().catch(error=>{$("#workspace-content").innerHTML=`<div class="notice error">${esc(error.message)}</div>`});
