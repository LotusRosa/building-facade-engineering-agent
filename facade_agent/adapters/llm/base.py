from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMConfig:
    mode: str = "disabled"
    provider: str = "custom"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key: str = ""
    protocol: str = "responses"
    timeout_seconds: float = 60.0
    verified: bool = False

    @property
    def configured(self) -> bool:
        return self.mode == "openai_compatible" and bool(self.base_url and self.model)

    def public_status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "protocol": self.protocol,
            "configured": self.configured,
            "verified": self.verified,
            "api_key_present": bool(self.api_key),
        }


@dataclass(frozen=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    text: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)


class ModelClient(Protocol):
    config: LLMConfig

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn: ...
