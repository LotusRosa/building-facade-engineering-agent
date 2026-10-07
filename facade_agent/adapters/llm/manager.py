from __future__ import annotations

import os
import hashlib
import json
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from .base import LLMConfig, ModelClient
from .openai_compatible import OpenAICompatibleClient
from .presets import get_provider_preset, list_provider_presets


class LLMManager:
    """Process-local provider configuration. API keys are never persisted."""

    def __init__(self, client_factory=OpenAICompatibleClient) -> None:
        self._lock = threading.RLock()
        self._client_factory = client_factory
        self._verification_tokens: dict[str, tuple[str, float]] = {}
        self._config = self._from_environment()

    @staticmethod
    def _from_environment() -> LLMConfig:
        mode = os.getenv("FACADE_LLM_MODE", "disabled").strip().lower()
        return LLMConfig(
            mode=mode,
            provider=os.getenv("FACADE_LLM_PROVIDER", "custom").strip().lower(),
            base_url=os.getenv("FACADE_LLM_BASE_URL", "https://api.openai.com/v1").strip(),
            model=os.getenv("FACADE_LLM_MODEL", "").strip(),
            api_key=os.getenv("FACADE_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip(),
            protocol=os.getenv("FACADE_LLM_PROTOCOL", "responses").strip().lower(),
            timeout_seconds=float(os.getenv("FACADE_LLM_TIMEOUT", "60")),
            verified=False,
        )

    @staticmethod
    def presets() -> list[dict[str, Any]]:
        return list_provider_presets()

    @staticmethod
    def _fingerprint(config: LLMConfig) -> str:
        material = json.dumps(
            {
                "mode": config.mode,
                "provider": config.provider,
                "base_url": config.base_url,
                "model": config.model,
                "api_key": config.api_key,
                "protocol": config.protocol,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_base_url(value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Base URL must be a complete http:// or https:// address.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Base URL cannot contain credentials, a query, or a fragment.")
        if "api_keys" in parsed.path.lower() or "api-keys" in parsed.path.lower():
            raise ValueError("Base URL points to a key-management webpage, not an API endpoint.")
        return value.rstrip("/")

    def _build_config(self, values: dict[str, Any], *, verified: bool = False) -> LLMConfig:
        mode = str(values.get("mode", "disabled")).strip().lower()
        if mode not in {"disabled", "openai_compatible"}:
            raise ValueError("Mode must be disabled or openai_compatible.")
        provider = str(values.get("provider", "custom")).strip().lower()
        preset = get_provider_preset(provider)
        protocol = str(values.get("protocol", "responses")).strip().lower()
        if protocol not in {"responses", "chat_completions"}:
            raise ValueError("Protocol must be responses or chat_completions.")
        base_url = str(values.get("base_url", "")).strip()
        model = str(values.get("model", "")).strip()
        api_key = str(values.get("api_key", "")).strip()
        if mode == "disabled":
            return LLMConfig(mode="disabled", provider=provider)
        if not base_url or not model:
            raise ValueError("Base URL and model are required when the LLM is enabled.")
        base_url = self._validate_base_url(base_url)
        if preset["base_url_locked"] and base_url != preset["base_url"]:
            raise ValueError("Base URL must match the selected provider preset.")
        if preset["base_url_locked"] and protocol != preset["protocol"]:
            raise ValueError("Protocol must match the selected provider preset.")
        if preset["api_key_required"] and not api_key:
            raise ValueError("An API key is required for the selected provider.")
        return LLMConfig(
            mode=mode,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            protocol=protocol,
            timeout_seconds=60.0,
            verified=verified,
        )

    def test_connection(self, values: dict[str, Any]) -> dict[str, Any]:
        config = self._build_config(values)
        client = self._client_factory(config)
        client.complete(
            messages=[
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "Connection test."},
            ],
            tools=[],
        )
        token = uuid.uuid4().hex
        with self._lock:
            now = time.monotonic()
            self._verification_tokens = {
                key: item for key, item in self._verification_tokens.items()
                if now - item[1] <= 300
            }
            self._verification_tokens[token] = (self._fingerprint(config), now)
        return {"verified": True, "verification_token": token}

    def configure(self, values: dict[str, Any]) -> dict[str, Any]:
        candidate = self._build_config(values)
        if candidate.mode == "disabled":
            with self._lock:
                self._config = candidate
                self._verification_tokens.clear()
                return self._config.public_status()
        token = str(values.get("verification_token", "")).strip()
        if not token:
            raise ValueError("Please test the connection before connecting the model.")
        with self._lock:
            verified = self._verification_tokens.pop(token, None)
            if verified is None:
                raise ValueError("The connection test has expired. Please test again.")
            if time.monotonic() - verified[1] > 300:
                raise ValueError("The connection test has expired. Please test again.")
            if verified[0] != self._fingerprint(candidate):
                raise ValueError("The model settings changed after testing. Please test again.")
            self._config = LLMConfig(
                mode=candidate.mode,
                provider=candidate.provider,
                base_url=candidate.base_url,
                model=candidate.model,
                api_key=candidate.api_key,
                protocol=candidate.protocol,
                timeout_seconds=candidate.timeout_seconds,
                verified=True,
            )
            return self._config.public_status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._config.public_status()

    def client(self) -> ModelClient | None:
        with self._lock:
            if not self._config.configured or not self._config.verified:
                return None
            return self._client_factory(self._config)
