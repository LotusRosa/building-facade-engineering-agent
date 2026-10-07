from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    source: tuple[str, ...]
    target: str
    requires_confirmation: bool = False


TRANSITIONS = {
    "create_project": Transition(("EMPTY",), "PROJECT_CREATED"),
    "confirm_classes": Transition(("PROJECT_CREATED",), "CLASSES_CONFIRMED", True),
    "validate_data": Transition(("CLASSES_CONFIRMED",), "DATA_VALIDATED"),
    "freeze_labels": Transition(("DATA_VALIDATED",), "LABELS_FROZEN", True),
    "train_champion": Transition(("LABELS_FROZEN",), "CHAMPION_READY", True),
    "prepare_update": Transition(("CHAMPION_READY", "DEPLOYED", "HELD"), "UPDATE_LABELS_FROZEN", True),
    "discover_failures": Transition(("UPDATE_LABELS_FROZEN",), "FAILURE_SLICES_READY"),
    "freeze_review": Transition(("FAILURE_SLICES_READY",), "REVIEW_FROZEN", True),
    "train_challenger": Transition(("REVIEW_FROZEN",), "CHALLENGER_READY", True),
    "build_evidence": Transition(("CHALLENGER_READY",), "EVIDENCE_READY"),
    "deploy": Transition(("EVIDENCE_READY",), "DEPLOYED", True),
    "hold": Transition(("EVIDENCE_READY",), "HELD", True),
}


def validate_transition(state: str, action: str, confirmed: bool) -> Transition:
    if action not in TRANSITIONS:
        raise ValueError(f"未知工具：{action}")
    transition = TRANSITIONS[action]
    if state not in transition.source:
        expected = " / ".join(transition.source)
        raise PermissionError(f"当前状态为 {state}，不能执行该操作；需要先进入 {expected}。")
    if transition.requires_confirmation and not confirmed:
        raise PermissionError("这是关键操作，需要人工确认后才能执行。")
    return transition

