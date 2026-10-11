from __future__ import annotations

import json
from typing import Any

from ..adapters.llm import LLMManager, ModelClient


SYSTEM_PROMPT = """You are Building-Facade Engineering Agent, a governed conversational assistant for civil engineers maintaining an image-level multi-label visual screening model.

Rules:
1. Explain in the user's language and use plain civil-engineering language.
2. Use only the supplied typed tools for state changes or calculations. Never invent metrics, file counts, hashes, model results, or successful actions.
3. The task is image-level multi-label screening, not bounding-box localization. Never claim localization.
4. The class named hollow means visually suspected hollowing and is not a substitute for field confirmation.
5. Never select labels unless the engineer explicitly states them for the current image.
6. Critical operations require human confirmation. You may propose them, but the controller decides whether they execute.
7. Never decide Deploy or Hold for the engineer.
8. Do not request, reveal, or repeat API keys.
9. If the current state does not permit an action, explain the required preceding step.
10. Keep responses concise and actionable.
11. Never propose or alter training preset fields, seeds, GPU command arguments, or result values. The deterministic backend owns them.
12. Failure Slice review belongs to one human domain expert. You may explain Accept, Trim, and Reject, but you may not submit a review, alter labels, or request reclustering.
13. Challenger update pools, draw counts, seeds, epochs, and sampling are owned by the versioned backend profile. Never propose overrides.
14. Treat conversation as the primary interface. When the user's intent and required arguments are clear, use the governed tool instead of directing them to a separate workbench.
15. Local file selection, image-label confirmation, Failure Slice review, and Deploy/Hold remain explicit human interactions. Explain the required interaction without pretending the model performed it.
16. Discovery data from every completed Deploy or Hold round becomes eligible training history. Gate data is immutable evaluation-only history and must never be proposed for training. The backend, not the language model, determines the complete accumulated Gate lineage.
17. Never invent, infer, or submit a local model-storage path. For a new project, collect only the project name and fixed defect classes, then ask the engineer to confirm the models-only folder in the task panel.
18. Training and maintenance data can be imported from an image folder with an optional .xlsx or UTF-8 .csv annotation table. The project template has filename, one 0/1 column per frozen class, and no_defect. Multiple defects may coexist; no_defect is exclusive. Blank label cells mean unannotated, not a negative sample. Complete verified tables skip manual annotation, but never bypass dataset checks, engineer confirmation, or training permissions. Direct local selection and table preview to the Task Panel; never ask the engineer to prepare a ZIP for training-data input.
19. Before suggesting annotation, read annotation_progress or get_annotation_status and report total, annotated, and remaining counts. If remaining labels exist, ask whether the engineer wants to import an existing table or annotate inside the Agent. If all labels are complete, offer validation instead of repeat annotation. Completeness is not the same as integrity validation or training readiness.
20. Every round needs separate maintenance/update (Train) data for screening and update training. Reserve an independent initial Core Safety seed once, separate from initial Train. Recommend Train:initial Core Safety = 7:3 or 8:2 at initialization, and Train:Current Gate = 7:3 or 8:2 for each maintenance round. These are optional guidance, not an enforced or automatic partition. Preserve class coverage, sufficient evaluation samples, and acquisition/building continuity when splitting; exact hashes cannot detect near-duplicate leakage. When an Active Champion exists, continue maintenance rather than reinitializing it. The first paired evaluation requires the reserved Core Safety seed and a fresh Current Gate; later rounds supply only Train and a fresh Current Gate, with cumulative Core Safety attached automatically. After successful evaluation and the human Retain/Promote decision, the completed Gate joins Core Safety. Engineers may prepare and annotate the required folders early; evaluation jobs still require a registered Challenger and human confirmation. Never send evaluation images into training.
"""


class AgentRuntime:
    def __init__(self, store: Any, registry: Any, llm_manager: LLMManager, workflow: Any | None = None) -> None:
        self.store = store
        self.registry = registry
        self.llm_manager = llm_manager
        self.workflow = workflow

    @staticmethod
    def _is_continue_intent(text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        return normalized in {"继续", "下一步", "继续下一步", "continue", "next", "go on"}

    @staticmethod
    def _data_split_guidance(*, initial: bool, english: bool = False) -> str:
        if english:
            roles = "Train : initial Core Safety" if initial else "Train : Current Gate"
            return f" Suggested {roles} = 7:3 or 8:2, not mandatory. Reserve the initial Core Safety once; later Gates accumulate automatically after completed evaluation and a human decision. Keep class coverage and related building/acquisition images together; evaluation images never enter Train."
        roles = "训练集 : 初始安全评估集" if initial else "训练集 : 本轮评估集"
        return f"建议 {roles}＝7:3 或 8:2，不强制。初始安全评估集仅预留一次；后续本轮评估集在完成评估和人工决定后自动累计。请保证类别覆盖，并把同一建筑/连续采集的相关图片放在同组，评估图片不能进入训练集。"

    @staticmethod
    def _annotation_message(progress: dict[str, Any], *, english: bool = False) -> str:
        cohort_titles = {'current_gate': '本轮评估集', 'core_safety': '初始安全评估集'}
        total, labeled, missing = (progress.get(k, 0) for k in ("image_count", "labeled_count", "unlabeled_count"))
        if progress.get("kind") == "evaluation":
            cohorts = progress["cohorts"]
            counts = " · ".join(f"{name if english else {'current_gate': '本轮评估集', 'core_safety': '初始安全评估集'}.get(name, name)}: {c['labeled_count']}/{c['image_count']}" for name, c in cohorts.items())
            if progress.get("dataset_status") == "frozen":
                return counts + (". Evaluation input is frozen; continue through the governed workflow." if english else "。评估数据已冻结，请按当前工作流继续。")
            return counts + (". Import an Excel/CSV table or annotate inside the Agent. Every required new evaluation cohort must be nonempty and fully annotated. Existing Core Safety is attached automatically; Train is separate." if english else "。可导入 Excel/CSV 表格或在智能助手内标注；本轮需提供的评估数据必须非空且标注完整。已有安全评估集自动附加；训练集另行准备。")
        if english:
            counts = f"Current dataset: {total} images, {labeled} annotated, {missing} unannotated. "
            if progress.get("evaluation_cohorts"):
                counts += "Independent evaluation labels: " + ", ".join(f"{name} {c['labeled_count']}/{c['image_count']}" for name, c in progress["evaluation_cohorts"].items()) + ". These folders never enter training. "
            if not total:
                return counts + "Select an image folder in the Task Panel, with an optional annotation table."
            if missing:
                return counts + "Would you like to import an existing Excel/CSV table or annotate the remaining images inside the Agent?"
            return counts + ("Labels are complete but still need validation before training preparation." if progress.get("dataset_status") == "open" else "Labels are complete and validated; repeat annotation is unnecessary. Continue through the governed workflow.")
        counts = f"当前数据集共 {total} 张，已标注 {labeled} 张，未标注 {missing} 张。"
        if progress.get("evaluation_cohorts"):
            counts += "独立评估标注：" + "、".join(f"{cohort_titles.get(name, name)} {c['labeled_count']}/{c['image_count']}" for name, c in progress["evaluation_cohorts"].items()) + "；评估文件夹不参与训练。"
        if not total:
            return counts + "请在任务面板选择图片文件夹，可同时提供已有标注表格。"
        if missing:
            return counts + "你要导入已有的 Excel/CSV 标注表格，还是在智能助手内标注剩余图片？"
        return counts + ("标注已经完整，不必重复标注；请确认数据校验后继续准备训练。" if progress.get("dataset_status") == "open" else "标签已完整且通过校验，不必重复标注；可按当前工作流继续。")

    @staticmethod
    def _action_title(action: dict[str, Any], *, english: bool) -> str:
        if english:
            return action['title']
        return {
            'start_next_round': '开始下一轮维护', 'start_first_round': '创建维护批次',
            'import_maintenance_data': '导入并标注维护数据', 'review_maintenance_labels': '检查维护数据标注',
            'start_failure_discovery': '启动当前模型筛查与失败发现', 'review_failure_slices': '完成专家失败切片审核',
            'register_challenger': '注册候选模型', 'start_challenger_training': '启动单 GPU 候选模型训练',
            'create_challenger_job': '生成候选模型训练任务', 'start_paired_evaluation': '启动当前模型与候选模型评估',
            'import_evaluation_snapshot': '导入本轮评估数据', 'record_engineer_decision': '选择保留当前模型或启用候选模型',
            'confirm_taxonomy': '确认并锁定类别', 'import_initial_data': '导入并标注初始训练数据',
            'validate_initial_labels': '检查完整标注并冻结数据集', 'review_initial_labels': '导入标签表格或完成逐图标注',
            'start_initial_training': '启动单 GPU 初始模型训练', 'prepare_initial_training': '生成初始模型训练任务',
            'register_initial_champion': '注册初始模型',
            'validate_maintenance_labels': '校验完整维护标注', 'freeze_maintenance_labels': '确认并冻结维护标注',
        }.get(action['action_id'], '完成当前步骤')

    @staticmethod
    def _english(language: str | None) -> bool:
        if language not in (None, 'en', 'zh'):
            raise ValueError('Language must be en or zh.')
        return language == 'en'

    def _workflow_response(self, project_id: str, provider: dict[str, Any], *, english: bool = False) -> dict[str, Any]:
        snapshot = self.workflow.get(project_id)
        queue_items = snapshot.get("gpu_queue", {}).get("project_items", [])
        if queue_items:
            item = queue_items[0]
            position = item.get("queue_position")
            if item.get("status") == "queued":
                message = f"GPU job queued; queue position: {position}." if english else f"GPU 任务已排队，当前位置：{position}。"
                status = "queued"
            elif item.get("status") == "starting":
                message = "GPU job is starting." if english else "GPU 任务正在启动。"
                status = "starting"
            else:
                message = "GPU job is running." if english else "GPU 任务正在运行。"
                status = "running"
            return {
                "status": status,
                "message": message,
                "queue": item,
                "workflow": snapshot,
                "provider": provider,
            }
        actions = snapshot["next_actions"]
        if not actions:
            return {"status": "completed", "message": "No pending workflow steps." if english else "当前没有待执行步骤。", "workflow": snapshot, "provider": provider}
        action = actions[0]
        title = self._action_title(action, english=english)
        if action["interaction_mode"] == "task_panel_required":
            message = (self._annotation_message(snapshot.get("annotation_progress", {}), english=english) if action["action_id"] in {"review_initial_labels", "review_maintenance_labels", "import_evaluation_snapshot", "import_maintenance_data"} else (f"Next: {title}. Opened the Task Panel." if english else f"下一步是：{title}。已定位到任务面板。"))
            if action["action_id"] in {"import_initial_data", "import_maintenance_data", "start_first_round", "start_next_round"}:
                message += self._data_split_guidance(initial=action["action_id"] == "import_initial_data", english=english)
            navigation = {
                "target": action["task_panel_target"],
                "action_id": action["action_id"],
                "workflow_version": snapshot["workflow_version"],
            }
            return {
                "status": "requires_human_input",
                "message": message,
                "navigation": navigation,
                "workflow": snapshot,
                "provider": provider,
            }
        pending = self.store.create_pending_tool_call(
            project_id=project_id,
            tool_name=action["tool_name"],
            arguments=action["arguments"],
            provider="deterministic_workflow",
            model="workflow-core",
            workflow_version=snapshot["workflow_version"],
        )
        return {
            "status": "awaiting_human_confirmation",
            "message": f"Ready: {title}. Confirm before execution." if english else f"已准备：{title}。请确认后执行。",
            "pending": pending,
            "workflow": snapshot,
            "provider": provider,
        }

    @staticmethod
    def _offline_reply(project: dict[str, Any] | None, *, english: bool = False) -> str:
        state = project["state"] if project else "EMPTY"
        if english:
            active_batch = next((b for b in (project or {}).get('maintenance_batches', []) if b['batch_id'] == project.get('active_batch_id')), None) if project else None
            batch_replies = {
                'MAINTENANCE_BATCH_FROZEN': 'Maintenance data is frozen. Confirm Champion screening and failure discovery; this does not train or deploy a model.',
                'FAILURE_DISCOVERY_COMPLETED': 'Verified Failure Slices await expert Accept / Trim / Reject review. They have not entered Challenger training.',
                'FAILURE_REVIEW_COMPLETED': 'Expert review is frozen. Confirm generating the governed Challenger job; training and deployment do not start automatically.',
            }
            if active_batch and active_batch['state'] in batch_replies:
                return batch_replies[active_batch['state']]
            return {
                'EMPTY': 'No language model is connected. Click Connect model at the top right to create and advance projects through conversation.',
                'PROJECT_CREATED': 'No language model is connected. Connect a model at the top right; an engineer must explicitly confirm the classes.',
                'TAXONOMY_CONFIRMED': 'Classes are frozen. Create a dataset and import local images in the Task Panel.',
                'DATA_IMPORTED': 'Import an Excel/CSV label table or annotate images in the Task Panel. Complete validated labels skip manual annotation; training still requires confirmation.',
                'DATA_VALIDATED': 'The dataset is frozen. Generate the Initial Champion training job next.',
                'INITIAL_TRAINING_READY': 'The Initial Champion job is ready. Confirm the GPU worker; verified results require a separate registration confirmation.',
                'CHAMPION_READY': 'Initial Champion is registered with verified weights, results, and audit records. Prepare a maintenance round next.',
            }.get(state, f'No language model is connected. Current governed state: {state}.')
        if project and state == "SCREENING_READY":
            active_batch = next(
                (
                    item for item in project.get("maintenance_batches", [])
                    if item["batch_id"] == project.get("active_batch_id")
                ),
                None,
            )
            if active_batch:
                batch_replies = {
                    "MAINTENANCE_BATCH_FROZEN": "维护批次已冻结。可由工程师确认启动一次 Champion 筛查与 Failure Discovery；系统不会调阈值、训练或部署。",
                    "FAILURE_DISCOVERY_COMPLETED": "候选 Failure Slices 已校验并等待专家 Accept / Trim / Reject；它们尚未进入 Challenger 训练。",
                    "FAILURE_REVIEW_COMPLETED": "专家审核已冻结为不可变知识版本。工程师可确认生成固定 160 failure + 160 remaining-new + 320 history-replay 的 Challenger 任务包；这一步尚不训练或部署模型。",
                }
                if active_batch["state"] in batch_replies:
                    return batch_replies[active_batch["state"]]
        replies = {
            "EMPTY": "语言模型尚未配置。请点击右上角“接入模型”；连接后即可通过自然语言创建和推进项目。",
            "PROJECT_CREATED": "语言模型尚未配置。请点击右上角“接入模型”；类别锁定仍需工程师明确确认。",
            "TAXONOMY_CONFIRMED": "语言模型尚未配置。当前可以创建数据集并导入本地图片。",
            "DATA_IMPORTED": "可以在任务面板导入已标注的 Excel/CSV 表格，或直接逐图多选标注。完整标签校验通过后可跳过人工标注；训练仍需确认。",
            "DATA_VALIDATED": "语言模型尚未配置。数据已冻结，下一阶段可以生成 Initial Champion 训练任务。",
            "INITIAL_TRAINING_READY": "Initial Champion 训练任务包已生成。请由工程师确认启动本机 GPU Worker；结果通过三种子与哈希校验后，还需再次人工确认才能注册。",
            "CHAMPION_READY": "Initial Champion 已注册并保留完整权重、结果和审计哈希。下一步可以准备筛查流程。",
        }
        return replies.get(state, f"语言模型尚未配置。当前受控状态为 {state}。")

    def _context(
        self, project_id: str | None, selected_image_id: str | None
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        project = self.store.get_project(project_id) if project_id else None
        context: dict[str, Any] = {
            "project": None,
            "selected_image": None,
            "privacy": "No image bytes are included in this language-model request.",
        }
        if project:
            datasets = self.store.list_datasets(project_id)
            context["project"] = {
                "project_id": project["project_id"],
                "name": project["name"],
                "state": project["state"],
                "classes": [
                    {"class_id": item["class_id"], "name": item["display_name"]}
                    for item in project["classes"]
                ],
                "datasets": [
                    {
                        "dataset_id": item["dataset_id"],
                        "name": item["name"],
                        "role": item["role"],
                        "status": item["status"],
                        "image_count": item["image_count"],
                        "labeled_count": item["labeled_count"],
                    }
                    for item in datasets
                ],
                "active_maintenance_batch": next(
                    (
                        {
                            "batch_id": item["batch_id"],
                            "name": item["name"],
                            "state": item["state"],
                            "dataset_id": item.get("dataset_id"),
                            "dataset_status": item.get("dataset_status"),
                        }
                        for item in project.get("maintenance_batches", [])
                        if item["batch_id"] == project.get("active_batch_id")
                    ),
                    None,
                ),
            }
        if selected_image_id:
            image = self.store.get_image(selected_image_id)
            if project and image["dataset_id"] not in {
                item["dataset_id"] for item in self.store.list_datasets(project_id)
            }:
                raise PermissionError("Selected image does not belong to the active project.")
            context["selected_image"] = {
                "image_id": image["image_id"],
                "dataset_id": image["dataset_id"],
                "annotation_status": image["annotation_status"],
                "class_ids": image["class_ids"],
                "no_defect": bool(image["no_defect"]),
            }
        return project, context

    def chat(
        self,
        *,
        text: str,
        project_id: str | None,
        selected_image_id: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        english = self._english(language)
        user_text = str(text).strip()
        if not user_text:
            raise ValueError("A message is required.")
        if len(user_text) > 8000:
            raise ValueError("Message is too long.")
        project, context = self._context(project_id, selected_image_id)
        status = self.llm_manager.status()
        provider = status["mode"]
        model = status["model"] or "disabled"
        user_message = self.store.record_llm_message(
            project_id=project_id,
            role="user",
            content=user_text,
            provider=provider,
            model=model,
            metadata={"selected_image_id": selected_image_id},
        )
        if project_id and self.workflow is not None:
            context["workflow"] = self.workflow.get(project_id)
            normalized = user_text.strip().casefold().rstrip("?？。!")
            annotation_queries = {"标注情况", "标注状态", "查看标注情况", "检查标注", "标注完成了吗", "需要标注吗", "还需要标注吗", "annotation status", "label status", "are labels complete", "do i need to annotate"}
            if normalized in annotation_queries:
                progress = context["workflow"]["annotation_progress"]
                result = {"status": "completed", "message": self._annotation_message(progress, english=english if language else normalized.isascii()), "annotation_progress": progress, "provider": status}
                self.store.record_llm_message(project_id=project_id, role="assistant", content=result["message"], provider="deterministic_workflow", model="workflow-core", metadata={"annotation_progress": progress})
                return result
            if self._is_continue_intent(user_text):
                result = self._workflow_response(project_id, status, english=english)
                self.store.record_llm_message(
                    project_id=project_id,
                    role="assistant",
                    content=result["message"],
                    provider="deterministic_workflow",
                    model="workflow-core",
                    metadata={key: value for key, value in result.items() if key not in {"message", "provider"}},
                )
                return result
        client = self.llm_manager.client()
        if client is None:
            reply = self._offline_reply(project, english=english)
            self.store.record_llm_message(
                project_id=project_id,
                role="assistant",
                content=reply,
                provider="disabled",
                model="disabled",
            )
            return {"status": "disabled", "message": reply, "provider": status}

        history = self.store.list_llm_messages(project_id, limit=16)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + (f"\nThe interface language is {language}. All assistant replies must use {'English' if english else 'Chinese'}, while preserving user-defined names, filenames, and identifiers." if language else '') + "\nCurrent deterministic context:\n" + json.dumps(context, ensure_ascii=False),
            }
        ]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item["role"] in {"user", "assistant"}
        )
        tools = self.registry.describe()
        deterministic_results: list[dict[str, Any]] = []
        seen_tool_calls: set[str] = set()
        conversation_project_id = project_id
        last_text = ""
        for _ in range(3):
            turn = client.complete(messages=messages, tools=tools)
            last_text = turn.text or last_text
            if not turn.tool_calls:
                reply = turn.text or ("Operation completed." if english else "操作已完成。")
                self.store.record_llm_message(
                    project_id=conversation_project_id,
                    role="assistant",
                    content=reply,
                    provider=client.config.mode,
                    model=client.config.model,
                    metadata={"tool_results": deterministic_results},
                )
                return {
                    "status": "completed",
                    "message": reply,
                    "tool_results": deterministic_results,
                    "provider": client.config.public_status(),
                }

            call = turn.tool_calls[0]
            if call.name == "create_project":
                project_name = call.arguments.get("project_name")
                classes = call.arguments.get("classes")
                draft = {
                    "project_name": project_name if isinstance(project_name, str) else "",
                    "classes": (
                        [item for item in classes if isinstance(item, str)]
                        if isinstance(classes, list)
                        else []
                    ),
                }
                reply = ("Project name and classes are ready. Enter and confirm the model storage folder in the Task Panel. The Agent never invents local paths; this folder stores only registered, verified models." if english else (
                    "项目名称和缺陷类别已整理好。请在已打开的任务面板中由工程师填写并确认"
                    "“模型保存文件夹”；该目录只保存训练完成且注册成功的模型，Agent 不会代填本机路径。"
                ))
                self.store.record_llm_message(
                    project_id=conversation_project_id,
                    role="assistant",
                    content=reply,
                    provider=client.config.mode,
                    model=client.config.model,
                    metadata={"action": "open_project_creation", "project_draft": draft},
                )
                return {
                    "status": "requires_human_input",
                    "action": "open_project_creation",
                    "message": reply,
                    "project_draft": draft,
                    "provider": client.config.public_status(),
                }
            call_signature = json.dumps(
                {"name": call.name, "arguments": call.arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if call_signature in seen_tool_calls:
                reply = turn.text or ("Repeated tool request blocked; the first result is preserved." if english else "模型重复请求了同一操作；已保留首次结果并停止重复执行。")
                self.store.record_llm_message(
                    project_id=conversation_project_id,
                    role="assistant",
                    content=reply,
                    provider=client.config.mode,
                    model=client.config.model,
                    metadata={"tool_results": deterministic_results, "repeat_blocked": True},
                )
                return {
                    "status": "completed",
                    "message": reply,
                    "tool_results": deterministic_results,
                    "repeat_blocked": True,
                    "provider": client.config.public_status(),
                }
            seen_tool_calls.add(call_signature)
            definition = self.registry.get(call.name)
            workflow_snapshot = self.workflow.get(conversation_project_id) if self.workflow and conversation_project_id else None
            current_action = next(
                (item for item in (workflow_snapshot or {}).get("next_actions", []) if item.get("tool_name") == call.name),
                None,
            )
            if (
                workflow_snapshot is not None
                and definition.policy.mutates_state
                and conversation_project_id
                and call.name != "create_project"
            ):
                if current_action is None:
                    raise PermissionError("This tool is not the current governed workflow action.")
                if current_action["interaction_mode"] == "task_panel_required":
                    return {
                        "status": "requires_human_input",
                        "message": (f"Next: {self._action_title(current_action, english=True)}. Complete it in the Task Panel." if english else f"下一步是：{self._action_title(current_action, english=False)}。请在任务面板完成。"),
                        "navigation": {
                            "target": current_action["task_panel_target"],
                            "action_id": current_action["action_id"],
                            "workflow_version": workflow_snapshot["workflow_version"],
                        },
                        "provider": client.config.public_status(),
                    }
                locked_arguments = current_action["arguments"]
                if any(call.arguments.get(key) != value for key, value in locked_arguments.items()):
                    raise PermissionError("Tool arguments differ from the authoritative workflow action.")
            if definition.policy.requires_confirmation:
                pending = self.store.create_pending_tool_call(
                    project_id=conversation_project_id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    provider=client.config.mode,
                    model=client.config.model,
                    workflow_version=(workflow_snapshot or {}).get("workflow_version"),
                )
                reply = turn.text or (f"Governed tool {call.name} is ready; engineer confirmation required." if english else f"已准备受控工具 {call.name}，等待工程师确认。")
                self.store.record_llm_message(
                    project_id=conversation_project_id,
                    role="assistant",
                    content=reply,
                    provider=client.config.mode,
                    model=client.config.model,
                    metadata={"pending_id": pending["pending_id"], "tool_name": call.name},
                )
                return {
                    "status": "awaiting_human_confirmation",
                    "message": reply,
                    "pending": pending,
                    "provider": client.config.public_status(),
                }

            try:
                result = self.registry.execute(
                    call.name,
                    call.arguments,
                    actor_type="llm",
                    actor_id=client.config.model,
                    confirmed=False,
                )
                if self.workflow is not None:
                    result = self.workflow.enrich_tool_result(conversation_project_id, result)
                run_id = self.store.record_tool_run(
                    project_id=project_id,
                    tool_name=call.name,
                    status="completed",
                    request=call.arguments,
                    response=result,
                )
                tool_result = {"tool": call.name, "ok": True, "result": result, "run_id": run_id}
            except Exception as error:
                run_id = self.store.record_tool_run(
                    project_id=project_id,
                    tool_name=call.name,
                    status="failed",
                    request=call.arguments,
                    response={"error": str(error)},
                )
                tool_result = {"tool": call.name, "ok": False, "error": str(error), "run_id": run_id}
            deterministic_results.append(tool_result)
            if (
                tool_result["ok"]
                and call.name == "create_project"
                and conversation_project_id is None
            ):
                conversation_project_id = tool_result["result"]["project"]["project_id"]
                self.store.move_llm_message_to_project(
                    user_message["message_id"], conversation_project_id
                )
            messages.append({"role": "assistant", "content": turn.text or f"Requested tool: {call.name}"})
            messages.append(
                {
                    "role": "user",
                    "content": "Deterministic tool result (do not alter or invent values):\n" + json.dumps(tool_result, ensure_ascii=False),
                }
            )

        reply = last_text or ("Tool-call limit reached. Review the results before continuing." if english else "已达到本轮工具调用上限。请检查结果后继续。")
        self.store.record_llm_message(
            project_id=conversation_project_id,
            role="assistant",
            content=reply,
            provider=client.config.mode,
            model=client.config.model,
            metadata={"tool_results": deterministic_results, "tool_limit_reached": True},
        )
        return {
            "status": "tool_limit_reached",
            "message": reply,
            "tool_results": deterministic_results,
            "provider": client.config.public_status(),
        }

    def resolve_confirmation(
        self,
        *,
        pending_id: str,
        approved: bool,
        actor_id: str = "local_engineer",
        language: str | None = None,
    ) -> dict[str, Any]:
        english = self._english(language)
        pending = self.store.get_pending_tool_call(pending_id)
        if pending["status"] != "pending":
            raise PermissionError("This tool call has already been resolved.")
        if not approved:
            resolved = self.store.resolve_pending_tool_call(
                pending_id=pending_id,
                status="rejected",
                resolved_by=actor_id,
                result=None,
            )
            return {"status": "rejected", "message": "Engineer rejected this tool call." if english else "工程师已拒绝该工具调用。", "pending": resolved}
        if self.workflow is not None and pending["project_id"] and pending.get("workflow_version"):
            current_version = self.workflow.get(pending["project_id"])["workflow_version"]
            if current_version != pending["workflow_version"]:
                self.store.resolve_pending_tool_call(
                    pending_id=pending_id,
                    status="failed",
                    resolved_by=actor_id,
                    result={"error": "stale_workflow_confirmation"},
                )
                raise PermissionError("The workflow changed; request the current action again.")
        try:
            result = self.registry.execute(
                pending["tool_name"],
                pending["arguments"],
                actor_type="human",
                actor_id=actor_id,
                confirmed=True,
            )
            if self.workflow is not None:
                result = self.workflow.enrich_tool_result(pending["project_id"], result)
            self.store.record_tool_run(
                project_id=pending["project_id"],
                tool_name=pending["tool_name"],
                status="completed",
                request=pending["arguments"],
                response=result,
            )
            resolved = self.store.resolve_pending_tool_call(
                pending_id=pending_id,
                status="approved",
                resolved_by=actor_id,
                result=result,
            )
            message = (f"Engineer confirmed; governed tool {pending['tool_name']} completed." if english else f"工程师已确认，受控工具 {pending['tool_name']} 执行完成。")
            self.store.record_llm_message(
                project_id=pending["project_id"],
                role="assistant",
                content=message,
                provider=pending["provider"],
                model=pending["model"],
                metadata={"pending_id": pending_id, "approved": True},
            )
            return {"status": "approved", "message": message, "result": result, "pending": resolved}
        except Exception as error:
            self.store.record_tool_run(
                project_id=pending["project_id"],
                tool_name=pending["tool_name"],
                status="failed",
                request=pending["arguments"],
                response={"error": str(error)},
            )
            self.store.resolve_pending_tool_call(
                pending_id=pending_id,
                status="failed",
                resolved_by=actor_id,
                result={"error": str(error)},
            )
            raise
