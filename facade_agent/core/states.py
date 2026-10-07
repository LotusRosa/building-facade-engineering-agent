from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProjectState(StrEnum):
    PROJECT_CREATED = "PROJECT_CREATED"
    TAXONOMY_CONFIRMED = "TAXONOMY_CONFIRMED"
    DATA_IMPORTED = "DATA_IMPORTED"
    DATA_VALIDATED = "DATA_VALIDATED"
    ANNOTATION_READY = "ANNOTATION_READY"
    INITIAL_TRAINING_READY = "INITIAL_TRAINING_READY"
    CHAMPION_READY = "CHAMPION_READY"
    SCREENING_READY = "SCREENING_READY"


class BatchState(StrEnum):
    CREATED = "CREATED"
    MAINTENANCE_DATA_IMPORTED = "MAINTENANCE_DATA_IMPORTED"
    MAINTENANCE_LABELS_READY = "MAINTENANCE_LABELS_READY"
    MAINTENANCE_BATCH_FROZEN = "MAINTENANCE_BATCH_FROZEN"
    LABELS_CONFIRMED = "LABELS_CONFIRMED"
    INFERENCE_COMPLETED = "INFERENCE_COMPLETED"
    FAILURE_DISCOVERY_COMPLETED = "FAILURE_DISCOVERY_COMPLETED"
    FAILURE_REVIEW_COMPLETED = "FAILURE_REVIEW_COMPLETED"
    CHALLENGER_TRAINED = "CHALLENGER_TRAINED"
    CHALLENGER_EVALUATED = "CHALLENGER_EVALUATED"
    DECISION_PENDING = "DECISION_PENDING"
    DEPLOYED = "DEPLOYED"
    HELD = "HELD"


@dataclass(frozen=True)
class StateTransition:
    sources: tuple[str, ...]
    target: str
    requires_confirmation: bool = False


PROJECT_TRANSITIONS: dict[str, StateTransition] = {
    "confirm_taxonomy": StateTransition((ProjectState.PROJECT_CREATED,), ProjectState.TAXONOMY_CONFIRMED, True),
    "record_data_import": StateTransition((ProjectState.TAXONOMY_CONFIRMED,), ProjectState.DATA_IMPORTED),
    "record_data_validation": StateTransition((ProjectState.DATA_IMPORTED,), ProjectState.DATA_VALIDATED),
    "record_annotation_ready": StateTransition((ProjectState.DATA_VALIDATED,), ProjectState.ANNOTATION_READY, True),
    "record_initial_training_ready": StateTransition((ProjectState.ANNOTATION_READY,), ProjectState.INITIAL_TRAINING_READY, True),
    "register_initial_champion": StateTransition((ProjectState.INITIAL_TRAINING_READY,), ProjectState.CHAMPION_READY, True),
    "enable_screening": StateTransition((ProjectState.CHAMPION_READY,), ProjectState.SCREENING_READY),
}


BATCH_TRANSITIONS: dict[str, StateTransition] = {
    "record_maintenance_data_import": StateTransition(
        (BatchState.CREATED,), BatchState.MAINTENANCE_DATA_IMPORTED
    ),
    "record_maintenance_labels_ready": StateTransition(
        (BatchState.MAINTENANCE_DATA_IMPORTED,), BatchState.MAINTENANCE_LABELS_READY
    ),
    "freeze_maintenance_batch": StateTransition(
        (BatchState.MAINTENANCE_LABELS_READY,),
        BatchState.MAINTENANCE_BATCH_FROZEN,
        True,
    ),
    "confirm_batch_labels": StateTransition((BatchState.CREATED,), BatchState.LABELS_CONFIRMED, True),
    "record_batch_inference": StateTransition(
        (BatchState.MAINTENANCE_BATCH_FROZEN, BatchState.LABELS_CONFIRMED),
        BatchState.INFERENCE_COMPLETED,
    ),
    "record_failure_discovery": StateTransition((BatchState.INFERENCE_COMPLETED,), BatchState.FAILURE_DISCOVERY_COMPLETED),
    "freeze_failure_review": StateTransition((BatchState.FAILURE_DISCOVERY_COMPLETED,), BatchState.FAILURE_REVIEW_COMPLETED, True),
    "record_challenger_training": StateTransition((BatchState.FAILURE_REVIEW_COMPLETED,), BatchState.CHALLENGER_TRAINED, True),
    "record_challenger_evaluation": StateTransition((BatchState.CHALLENGER_TRAINED,), BatchState.CHALLENGER_EVALUATED),
    "open_human_decision": StateTransition((BatchState.CHALLENGER_EVALUATED,), BatchState.DECISION_PENDING),
    "deploy_challenger": StateTransition((BatchState.DECISION_PENDING,), BatchState.DEPLOYED, True),
    "hold_challenger": StateTransition((BatchState.DECISION_PENDING,), BatchState.HELD, True),
    "hold_without_challenger": StateTransition(
        (BatchState.FAILURE_REVIEW_COMPLETED,), BatchState.HELD, True
    ),
}


def require_transition(
    current_state: str,
    action: str,
    transitions: dict[str, StateTransition],
    confirmed: bool,
) -> StateTransition:
    try:
        transition = transitions[action]
    except KeyError as exc:
        raise ValueError(f"Unknown governed action: {action}") from exc
    if current_state not in transition.sources:
        expected = ", ".join(str(value) for value in transition.sources)
        raise PermissionError(
            f"Action {action} is not allowed from {current_state}; expected {expected}."
        )
    if transition.requires_confirmation and not confirmed:
        raise PermissionError("Human confirmation is required for this critical action.")
    return transition

