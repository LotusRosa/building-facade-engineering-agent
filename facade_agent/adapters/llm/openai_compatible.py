from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any

from .base import LLMConfig, ModelToolCall, ModelTurn


class LLMConnectionError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        if not config.configured:
            raise ValueError("The LLM provider is not configured.")
        if config.protocol not in {"responses", "chat_completions"}:
            raise ValueError("Protocol must be responses or chat_completions.")
        self.config = config

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            messages = {
                401: "API key is invalid or belongs to a different provider.",
                403: "The API key cannot access this model.",
                404: "The API endpoint or model name is incorrect.",
                429: "The provider rate limit or account quota was reached.",
            }
            raise LLMConnectionError(
                messages.get(exc.code, f"The model service returned HTTP {exc.code}.")
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMConnectionError("Unable to reach or parse the model service.") from exc

    @staticmethod
    def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
                "strict": True,
            }
            for tool in tools
        ]

    @staticmethod
    def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        if self.config.protocol == "chat_completions":
            return self._complete_chat(messages, tools)
        return self._complete_responses(messages, tools)

    def _complete_responses(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        system = "\n\n".join(item["content"] for item in messages if item["role"] == "system")
        inputs = [
            {"role": item["role"], "content": item["content"]}
            for item in messages
            if item["role"] in {"user", "assistant"}
        ]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "instructions": system,
            "input": inputs,
            "tools": self._responses_tools(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": False,
            "max_output_tokens": 1200,
        }
        data = self._post("responses", payload)
        texts: list[str] = []
        calls: list[ModelToolCall] = []
        for item in data.get("output", []):
            if item.get("type") == "function_call":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    raise LLMConnectionError("The model returned invalid tool arguments.") from exc
                calls.append(ModelToolCall(item.get("call_id") or uuid.uuid4().hex, item.get("name", ""), arguments))
            elif item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        texts.append(content["text"])
        return ModelTurn(text="\n".join(texts).strip(), tool_calls=calls)

    def _complete_chat(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": self._chat_tools(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0.2,
        }
        data = self._post("chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise LLMConnectionError("The LLM endpoint returned no choices.")
        message = choices[0].get("message") or {}
        calls = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise LLMConnectionError("The model returned invalid tool arguments.") from exc
            calls.append(ModelToolCall(item.get("id") or uuid.uuid4().hex, function.get("name", ""), arguments))
        return ModelTurn(text=(message.get("content") or "").strip(), tool_calls=calls)
