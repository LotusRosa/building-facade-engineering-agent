from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from facade_agent.adapters.llm.base import LLMConfig, ModelToolCall, ModelTurn
from facade_agent.adapters.llm.manager import LLMManager
from facade_agent.adapters.llm.openai_compatible import OpenAICompatibleClient
from facade_agent.adapters.llm.presets import list_provider_presets
from facade_agent.application.agent_runtime import AgentRuntime
from facade_agent.application.gpu_scheduler import GpuJobScheduler
from facade_agent.application.workflow import WorkflowSnapshotService
from facade_agent.storage import Store
from facade_agent.core.permissions import ToolPolicy
from facade_agent.tools import ToolDefinition, build_phase1_registry


class FakeManager:
    def __init__(self, client):
        self._client = client
    def client(self):
        return self._client
    def status(self):
        return self._client.config.public_status() if self._client else LLMConfig().public_status()


class FakeClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.config = LLMConfig(mode="openai_compatible", base_url="http://local/v1", model="fake-model", protocol="chat_completions")
    def complete(self, *, messages, tools):
        self.last_messages = messages
        self.last_tools = tools
        return self.turns.pop(0)


class StubOpenAIClient(OpenAICompatibleClient):
    def __init__(self, config, response):
        super().__init__(config)
        self.response = response
    def _post(self, endpoint, payload):
        self.endpoint = endpoint
        self.payload = payload
        return self.response


class ConnectionTestClient:
    def __init__(self, config):
        self.config = config

    def complete(self, *, messages, tools):
        return ModelTurn(text="OK")


class Phase3ALLMAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "agent.sqlite3")
        self.registry = build_phase1_registry(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_critical_llm_tool_waits_for_human_confirmation(self):
        project = self.store.create_project(project_name="Facade", class_names=["crack"])["project"]
        client = FakeClient([
            ModelTurn(tool_calls=[ModelToolCall("call-1", "confirm_taxonomy", {"project_id": project["project_id"]})])
        ])
        runtime = AgentRuntime(self.store, self.registry, FakeManager(client))
        response = runtime.chat(text="确认类别", project_id=project["project_id"])
        self.assertEqual(response["status"], "awaiting_human_confirmation")
        self.assertEqual(self.store.get_project(project["project_id"])["state"], "PROJECT_CREATED")
        resolved = runtime.resolve_confirmation(pending_id=response["pending"]["pending_id"], approved=True)
        self.assertEqual(resolved["status"], "approved")
        self.assertEqual(self.store.get_project(project["project_id"])["state"], "TAXONOMY_CONFIRMED")
        with self.assertRaises(PermissionError):
            runtime.resolve_confirmation(pending_id=response["pending"]["pending_id"], approved=True)

    def test_llm_project_creation_opens_human_task_panel_without_creating(self):
        client = FakeClient([
            ModelTurn(tool_calls=[ModelToolCall("call-1", "create_project", {"project_name": "Bridge", "classes": ["crack", "rust"]})]),
        ])
        runtime = AgentRuntime(self.store, self.registry, FakeManager(client))
        response = runtime.chat(text="创建桥梁项目，识别裂缝和锈蚀", project_id=None)
        self.assertEqual(response["status"], "requires_human_input")
        self.assertEqual(response["action"], "open_project_creation")
        self.assertEqual(response["project_draft"], {"project_name": "Bridge", "classes": ["crack", "rust"]})
        self.assertIn("任务面板", response["message"])
        self.assertIn("模型保存文件夹", response["message"])
        self.assertEqual(self.store.list_projects(), [])

        invented = FakeClient([
            ModelTurn(tool_calls=[ModelToolCall(
                "call-2",
                "create_project",
                {
                    "project_name": "Unsafe",
                    "classes": ["crack"],
                    "model_storage_root": r"C:\invented\models",
                },
            )]),
        ])
        blocked = AgentRuntime(self.store, self.registry, FakeManager(invented)).chat(
            text="帮我自动选目录", project_id=None
        )
        self.assertEqual(blocked["status"], "requires_human_input")
        self.assertNotIn("model_storage_root", blocked["project_draft"])
        self.assertEqual(self.store.list_projects(), [])

    def test_repeated_identical_tool_call_is_executed_only_once(self):
        calls = []
        self.registry.register(ToolDefinition(
            "echo_once",
            "A noncritical deterministic test tool.",
            {
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=False, requires_confirmation=False),
            lambda args, ctx: calls.append(args["value"]) or {"value": args["value"]},
        ))
        call = ModelToolCall(
            "call-1",
            "echo_once",
            {"value": "Bridge"},
        )
        client = FakeClient(
            [ModelTurn(tool_calls=[call]), ModelTurn(tool_calls=[call])]
        )
        runtime = AgentRuntime(self.store, self.registry, FakeManager(client))

        response = runtime.chat(text="创建桥梁项目", project_id=None)

        self.assertEqual(response["status"], "completed")
        self.assertTrue(response["repeat_blocked"])
        self.assertEqual(len(response["tool_results"]), 1)
        self.assertEqual(calls, ["Bridge"])

    def test_disabled_mode_is_honest_and_persists_no_secret(self):
        runtime = AgentRuntime(self.store, self.registry, FakeManager(None))
        response = runtime.chat(text="下一步是什么", project_id=None)
        self.assertEqual(response["status"], "disabled")
        messages = self.store.list_llm_messages(None)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])

    def test_delete_conversation_preserves_project_and_other_scopes(self):
        project = self.store.create_project(
            project_name="Facade", class_names=["crack"]
        )["project"]
        self.store.record_llm_message(
            project_id=None,
            role="user",
            content="global",
            provider="disabled",
            model="none",
        )
        self.store.record_llm_message(
            project_id=project["project_id"],
            role="assistant",
            content="project",
            provider="disabled",
            model="none",
        )

        self.assertEqual(self.store.delete_llm_messages(None), 1)
        self.assertEqual(self.store.list_llm_messages(None), [])
        self.assertEqual(len(self.store.list_llm_messages(project["project_id"])), 1)
        self.assertEqual(self.store.get_project(project["project_id"])["name"], "Facade")

        self.assertEqual(self.store.delete_llm_messages(project["project_id"]), 1)
        self.assertEqual(self.store.list_llm_messages(project["project_id"]), [])
        self.assertEqual(self.store.get_project(project["project_id"])["name"], "Facade")

    def test_responses_parser_extracts_text_and_function_call(self):
        config = LLMConfig(mode="openai_compatible", base_url="http://local/v1", model="test", protocol="responses")
        client = StubOpenAIClient(config, {"output": [
            {"type": "message", "content": [{"type": "output_text", "text": "准备执行。"}]},
            {"type": "function_call", "call_id": "c1", "name": "create_project", "arguments": '{"project_name":"X","classes":["crack"]}'},
        ]})
        turn = client.complete(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tools=self.registry.describe())
        self.assertEqual(turn.text, "准备执行。")
        self.assertEqual(turn.tool_calls[0].name, "create_project")
        self.assertEqual(client.endpoint, "responses")
        self.assertFalse(client.payload["store"])
        self.assertFalse(client.payload["parallel_tool_calls"])

    def test_common_provider_presets_expose_locked_official_endpoints(self):
        presets = {item["id"]: item for item in list_provider_presets()}

        self.assertEqual(
            set(presets),
            {"openai", "deepseek", "openrouter", "lmstudio", "ollama", "custom"},
        )
        self.assertEqual(presets["openai"]["base_url"], "https://api.openai.com/v1")
        self.assertEqual(presets["deepseek"]["base_url"], "https://api.deepseek.com")
        self.assertEqual(presets["openrouter"]["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(presets["lmstudio"]["base_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(presets["ollama"]["base_url"], "http://127.0.0.1:11434/v1")
        self.assertTrue(all(presets[name]["base_url_locked"] for name in presets if name != "custom"))
        self.assertFalse(presets["custom"]["base_url_locked"])

    def test_enabled_configuration_must_match_a_successfully_tested_connection(self):
        manager = LLMManager(client_factory=ConnectionTestClient)
        values = {
            "mode": "openai_compatible",
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-flash",
            "api_key": "secret-key",
            "protocol": "responses",
        }

        with self.assertRaisesRegex(ValueError, "test the connection"):
            manager.configure(values)

        result = manager.test_connection(values)
        status = manager.configure({**values, "verification_token": result["verification_token"]})

        self.assertTrue(status["configured"])
        self.assertTrue(status["verified"])
        self.assertEqual(status["provider"], "deepseek")
        self.assertNotIn("secret-key", repr(status))

    def test_changing_a_tested_model_invalidates_the_verification_token(self):
        manager = LLMManager(client_factory=ConnectionTestClient)
        values = {
            "mode": "openai_compatible",
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3:8b",
            "api_key": "",
            "protocol": "responses",
        }
        token = manager.test_connection(values)["verification_token"]

        with self.assertRaisesRegex(ValueError, "changed"):
            manager.configure({**values, "model": "qwen3:14b", "verification_token": token})

    def test_provider_preset_rejects_a_key_management_webpage_as_base_url(self):
        manager = LLMManager(client_factory=ConnectionTestClient)
        values = {
            "mode": "openai_compatible",
            "provider": "deepseek",
            "base_url": "https://platform.deepseek.com/api_keys",
            "model": "deepseek-flash",
            "api_key": "secret-key",
            "protocol": "responses",
        }

        with self.assertRaises(ValueError):
            manager.test_connection(values)

    def test_provider_http_error_does_not_expose_response_body_or_key(self):
        config = LLMConfig(
            mode="openai_compatible",
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-test",
            api_key="sk-visible-secret",
            protocol="responses",
        )
        client = OpenAICompatibleClient(config)
        error = urllib.error.HTTPError(
            config.base_url,
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"error":{"message":"Incorrect API key: sk-visible-secret"}}'),
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(Exception, "API key") as caught:
                client.complete(
                    messages=[{"role": "user", "content": "ping"}],
                    tools=[],
                )

        message = str(caught.exception)
        self.assertNotIn("sk-visible-secret", message)
        self.assertNotIn("Incorrect API key", message)

    def test_offline_reply_points_to_the_top_right_model_control(self):
        reply = AgentRuntime._offline_reply(None)

        self.assertIn("右上角", reply)
        self.assertIn("接入模型", reply)
        self.assertNotIn("右下角高级设置", reply)

    def test_queue_conversation_and_workflow_share_authoritative_position(self):
        project = self.store.create_project(
            project_name="Facade", class_names=["crack"]
        )["project"]
        scheduler = GpuJobScheduler(self.store, lease_owner="private-lease")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            scheduler.enqueue_in_transaction(
                db,
                project_id=project["project_id"],
                pipeline="initial_champion",
                run_id="run_conversation_queue",
                actor_type="human",
                actor_id="engineer",
            )
        workflow = WorkflowSnapshotService(self.store, scheduler)

        snapshot = workflow.get(project["project_id"])
        response = AgentRuntime(
            self.store, self.registry, FakeManager(None), workflow
        ).chat(text="继续", project_id=project["project_id"])

        workflow_item = snapshot["gpu_queue"]["project_items"][0]
        conversation_item = response["workflow"]["gpu_queue"]["project_items"][0]
        self.assertEqual(workflow_item["queue_id"], conversation_item["queue_id"])
        self.assertEqual(workflow_item["queue_position"], 1)
        self.assertEqual(workflow_item["queue_position"], conversation_item["queue_position"])
        serialized = json.dumps(response["workflow"])
        self.assertNotIn("private-lease", serialized)
        self.assertNotIn("sha256", serialized)


if __name__ == "__main__":
    unittest.main()
