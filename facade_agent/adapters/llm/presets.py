from __future__ import annotations

from typing import Any


_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "openai",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "protocol": "responses",
        "model_example": "gpt-5.2",
        "api_key_required": True,
        "base_url_locked": True,
    },
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "protocol": "responses",
        "model_example": "deepseek-flash",
        "api_key_required": True,
        "base_url_locked": True,
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "protocol": "chat_completions",
        "model_example": "provider/model",
        "api_key_required": True,
        "base_url_locked": True,
    },
    {
        "id": "lmstudio",
        "label": "LM Studio (local)",
        "base_url": "http://127.0.0.1:1234/v1",
        "protocol": "responses",
        "model_example": "loaded-model-id",
        "api_key_required": False,
        "base_url_locked": True,
    },
    {
        "id": "ollama",
        "label": "Ollama (local)",
        "base_url": "http://127.0.0.1:11434/v1",
        "protocol": "responses",
        "model_example": "qwen3:8b",
        "api_key_required": False,
        "base_url_locked": True,
    },
    {
        "id": "custom",
        "label": "Custom OpenAI-compatible",
        "base_url": "",
        "protocol": "responses",
        "model_example": "your-model-id",
        "api_key_required": False,
        "base_url_locked": False,
    },
)


def list_provider_presets() -> list[dict[str, Any]]:
    return [dict(item) for item in _PRESETS]


def get_provider_preset(provider: str) -> dict[str, Any]:
    for item in _PRESETS:
        if item["id"] == provider:
            return dict(item)
    raise ValueError("Unknown model provider preset.")
