from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassDefinition:
    class_id: str
    display_name: str
    display_name_zh: str | None = None
    display_name_en: str | None = None
    description: str = ""
    sort_order: int = 0


@dataclass(frozen=True)
class ProjectRecord:
    project_id: str
    name: str
    state: str
    active_batch_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MaintenanceBatchRecord:
    batch_id: str
    project_id: str
    name: str
    state: str
    formal_decision: str | None
    decision_reason: str | None
    created_at: str
    updated_at: str

