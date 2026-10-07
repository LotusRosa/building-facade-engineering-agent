from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPolicy:
    mutates_state: bool
    requires_confirmation: bool
    allowed_actor_types: tuple[str, ...] = ("human", "llm", "system")

    def authorize(self, actor_type: str, confirmed: bool) -> None:
        if actor_type not in self.allowed_actor_types:
            raise PermissionError(f"Actor type {actor_type!r} is not allowed.")
        if self.requires_confirmation and (actor_type != "human" or not confirmed):
            raise PermissionError(
                "This critical tool requires an explicit human confirmation."
            )

