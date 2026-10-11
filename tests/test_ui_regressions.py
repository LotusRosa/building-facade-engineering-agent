from __future__ import annotations

import unittest
from pathlib import Path


class UIRegressionTests(unittest.TestCase):
    def test_language_controls_file_captions_and_chat_requests(self):
        root = Path(__file__).resolve().parents[1] / 'facade_agent' / 'static'
        html = (root / 'index.html').read_text(encoding='utf-8-sig')
        controls = (root / 'file_controls.js').read_text(encoding='utf-8-sig')
        app = (root / 'app.js').read_text(encoding='utf-8-sig')
        locale = (root / 'localization.js').read_text(encoding='utf-8-sig')
        self.assertIn('/file_controls.js', html)
        self.assertIn('/localization.js', html)
        self.assertLess(html.index('/evaluation_input.js'), html.index('/localization.js'))
        self.assertIn('file_none:"No files selected"', controls)
        self.assertIn('file_none:"未选择文件"', controls)
        self.assertIn('MutationObserver', controls)
        self.assertIn('input.disabled', controls)
        self.assertIn('language:app.lang', app)
        self.assertIn('data-i18n="prepare_runtime"', html)
        self.assertIn('api_key_example', locale)
        self.assertNotIn('refresh().catch', (root / 'evaluation_input.js').read_text(encoding='utf-8-sig'))

    def test_conversation_is_primary_and_workbench_is_an_advanced_drawer(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        styles = (root / "styles.css").read_text(encoding="utf-8-sig")
        self.assertIn("Building-Facade Engineering Agent", html)
        self.assertNotIn("Building Facace Agent", html)
        self.assertNotIn('data-i18n="brand_sub"', html)
        self.assertIn('id="advanced-button"', html)
        self.assertIn('id="advanced-workspace"', html)
        self.assertIn('aria-hidden="true"', html)
        self.assertIn('id="workspace-close"', html)
        self.assertLess(html.index('class="agent-panel"'), html.index('id="advanced-button"'))
        self.assertIn("function setAdvancedOpen(open)", script)
        self.assertIn('$("#advanced-button").onclick', script)
        self.assertIn(".workspace.open", styles)
        self.assertIn(".advanced-button", styles)

    def test_sidebar_is_local_chat_history_and_workflow_moves_to_lower_settings(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn('id="conversation-history"', html)
        self.assertIn('id="workflow-settings-button"', html)
        self.assertIn('id="workflow-drawer"', html)
        self.assertIn('id="workflow-list"', html)
        self.assertNotIn('id="project-select"', html)
        self.assertIn("conversationHistory", script)
        self.assertIn("function renderConversationHistory()", script)
        self.assertIn("function setWorkflowOpen(open)", script)

    def test_conversations_are_deletable_without_project_deletion(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "facade_agent" / "static" / "app.js").read_text(
            encoding="utf-8-sig"
        )
        server = (root / "facade_agent" / "server.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('data-delete-conversation-id', script)
        self.assertIn('/api/agent/messages/delete', script)
        self.assertIn('delete_conversation_confirm', script)
        self.assertIn('parsed.path == "/api/agent/messages/delete"', server)
        self.assertNotIn('DELETE FROM projects', server)

    def test_project_deletion_requires_exact_name_confirmation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "facade_agent" / "static" / "index.html").read_text(
            encoding="utf-8-sig"
        )
        script = (root / "facade_agent" / "static" / "app.js").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('id="delete-project-dialog"', html)
        self.assertIn('id="delete-project-name"', html)
        self.assertIn('data-remove-project-id', script)
        self.assertIn('confirmation_name:name', script)
        self.assertIn('tool:"delete_project"', script)

    def test_environment_and_model_connection_stay_next_to_language(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        top_actions = html.split('<div class="top-actions">', 1)[1].split("</div>", 1)[0]
        self.assertIn('id="environment-button"', top_actions)
        self.assertIn('id="llm-settings-button"', top_actions)
        self.assertIn('id="language-button"', top_actions)
        self.assertLess(top_actions.index('id="environment-button"'), top_actions.index('id="llm-settings-button"'))
        self.assertLess(top_actions.index('id="llm-settings-button"'), top_actions.index('id="language-button"'))

    def test_model_dialog_separates_provider_test_and_verified_connect(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")

        self.assertIn('id="llm-provider"', html)
        self.assertIn('id="llm-test"', html)
        self.assertIn('id="llm-connect"', html)
        self.assertIn('id="llm-verification"', html)
        self.assertIn('/api/llm/presets', script)
        self.assertIn('/api/llm/test', script)
        self.assertIn('verification_token', script)

    def test_bottom_controls_have_distinct_project_and_task_responsibilities(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")

        self.assertIn('data-i18n="project_workflow"', html)
        self.assertIn('data-i18n="advanced_settings"', html)
        self.assertIn('id="workflow-drawer"', html)
        self.assertIn('id="advanced-workspace"', html)

    def test_project_creation_requires_a_models_only_folder_in_task_panel(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")

        self.assertIn('id="model-storage-status"', html)
        self.assertIn('id="model-storage-root"', script)
        self.assertIn('model_storage_root:modelStorageRoot', script)
        self.assertIn('confirmed:true', script)
        self.assertIn('open_project_creation', script)
        self.assertIn('model_storage_only', script)
        self.assertIn('model_storage_locked', script)
        self.assertNotIn('id="model-storage-root"', html.split('<dialog id="llm-dialog">', 1)[1])
        self.assertNotIn('id="model-storage-root"', html.split('<dialog id="environment-dialog">', 1)[1])

    def test_language_switch_does_not_clear_project_conversation(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "facade_agent"
            / "static"
            / "app.js"
        ).read_text(encoding="utf-8-sig")
        handler = next(
            line for line in script.splitlines()
            if line.startswith('$("#language-button").onclick=')
        )
        self.assertNotIn("app.messages=[]", handler)
        self.assertIn("render()", handler)

    def test_public_ui_defaults_to_english_but_keeps_language_switch(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        self.assertIn('localStorage.getItem("facade_lang")||"en"', script)
        self.assertIn('app.lang=app.lang==="zh"?"en":"zh"', script)
        self.assertIn('<html lang="en">', html)

    def test_history_and_training_views_are_wired(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn('id="history-view-button"', html)
        self.assertIn("/api/project/history", script)
        self.assertIn("create_initial_training_job", script)
        self.assertIn("start_initial_champion_training", script)
        self.assertIn("cancel_initial_champion_training", script)
        self.assertIn("register_initial_champion", script)
        self.assertIn("/api/training/run", script)
        self.assertIn("renderTrainingReady", script)
        self.assertIn("renderChampion", script)

    def test_maintenance_batch_flow_is_wired_without_bbox_annotation(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn("AGENT 2.2 · COMPLETE", html)
        self.assertIn("/api/maintenance/batches", script)
        self.assertIn('tool:"create_maintenance_batch"', script)
        self.assertIn('tool:"create_maintenance_dataset"', script)
        self.assertIn('tool:"freeze_maintenance_batch"', script)
        self.assertIn("MAINTENANCE_DATA_IMPORTED", script)
        self.assertIn("MAINTENANCE_LABELS_READY", script)
        self.assertIn("MAINTENANCE_BATCH_FROZEN", script)
        self.assertNotIn('type="file" accept="application/json"', script)

    def test_task_panels_offer_general_verified_bundle_import_without_fixed_counts(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        script = (root / "app.js").read_text(encoding="utf-8-sig")

        self.assertIn('"initial-labeled-bundle-form"', script)
        self.assertIn('"maintenance-labeled-bundle-form"', script)
        self.assertIn('/api/labeled-bundles/import', script)
        self.assertIn('accept="application/zip,.zip"', script)
        self.assertIn("importLabeledBundle", script)
        self.assertNotIn("file.size===150", script)
        self.assertNotIn("file.size===50", script)

    def test_server_has_no_fabricated_legacy_training_results(self) -> None:
        server = (
            Path(__file__).resolve().parents[1] / "facade_agent" / "server.py"
        ).read_text(encoding="utf-8-sig")
        self.assertNotIn("Champion演示任务已完成", server)
        self.assertNotIn('parsed.path == "/api/action"', server)
        self.assertNotIn('parsed.path == "/api/review"', server)

    def test_failure_discovery_flow_is_wired_and_single_pass(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn("AGENT 2.2 · COMPLETE", html)
        self.assertIn("/api/screening/runs", script)
        self.assertIn("/api/failure-slices", script)
        self.assertIn('tool:"start_champion_failure_discovery"', script)
        self.assertIn('tool:"cancel_champion_failure_discovery"', script)
        self.assertIn("FAILURE_DISCOVERY_COMPLETED", script)
        self.assertIn("thresholds not retuned", script.lower())

    def test_single_expert_failure_review_is_wired_without_label_editing(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn("AGENT 2.2 · COMPLETE", html)
        self.assertIn("/api/failure-review", script)
        self.assertIn('tool:"save_failure_slice_review"', script)
        self.assertIn('tool:"freeze_failure_slice_review"', script)
        self.assertIn("Accept ·", script)
        self.assertIn("Trim ·", script)
        self.assertIn("Reject ·", script)
        self.assertIn("FAILURE_REVIEW_COMPLETED", script)
        self.assertIn("标签未修改", script)
        self.assertIn("未重新聚类", script)

    def test_challenger_job_is_fixed_and_has_no_parameter_controls(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn("AGENT 2.2 · COMPLETE", html)
        self.assertIn("/api/challenger/jobs", script)
        self.assertIn('tool:"create_challenger_training_job"', script)
        self.assertIn("160 / 160 / 320", script)
        self.assertNotIn('id="challenger-epochs"', script)
        self.assertNotIn('id="challenger-learning-rate"', script)

    def test_challenger_worker_requires_separate_human_registration(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn("/api/challenger/runs", script)
        self.assertIn("/api/challenger/run", script)
        self.assertIn('tool:"start_challenger_training"', script)
        self.assertIn('tool:"cancel_challenger_training"', script)
        self.assertIn('tool:"register_challenger"', script)
        self.assertIn("CHALLENGER_TRAINED", script)
        self.assertIn("Active Champion is unchanged", script)

    def test_independent_evaluation_and_human_decision_are_wired(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        self.assertIn("/api/evaluation/inbox", script)
        self.assertIn("/api/evaluation/evidence", script)
        self.assertIn('tool:"create_model_evaluation_job"', script)
        self.assertIn('tool:"start_model_evaluation"', script)
        self.assertIn('tool:"cancel_model_evaluation"', script)
        self.assertIn('tool:"record_deployment_decision"', script)
        self.assertIn('id="record-early-hold"', script)
        self.assertIn("hold_without_challenger", script)
        self.assertIn("current_gate", script)
        self.assertIn("core_safety", script)
        self.assertNotIn("Final Test", script)
        self.assertNotIn("final_test", script)

    def test_gpu_queue_has_one_global_single_gpu_experience(self) -> None:
        root = Path(__file__).resolve().parents[1] / "facade_agent" / "static"
        html = (root / "index.html").read_text(encoding="utf-8-sig")
        script = (root / "app.js").read_text(encoding="utf-8-sig")
        styles = (root / "styles.css").read_text(encoding="utf-8-sig")

        self.assertIn('id="gpu-queue-banner"', html)
        self.assertIn('/api/gpu-queue', script)
        self.assertIn('function renderGpuQueueBanner()', script)
        self.assertIn('function scheduleGpuQueuePoll()', script)
        self.assertIn('function cancelQueuedTask()', script)
        self.assertIn('Cancel queued task', script)
        self.assertIn('Queue position', script)
        self.assertIn('Initial Champion training', script)
        self.assertIn('Failure discovery', script)
        self.assertIn('Challenger training', script)
        self.assertIn('Champion–Challenger evaluation', script)
        self.assertIn('.gpu-queue-banner', styles)
        self.assertNotIn('DDP', script)
        self.assertNotIn('world_size', script)
        self.assertNotIn('multi-GPU', script)
        self.assertNotIn('Final Test', script)
        self.assertNotIn('final_test', script)


if __name__ == "__main__":
    unittest.main()
