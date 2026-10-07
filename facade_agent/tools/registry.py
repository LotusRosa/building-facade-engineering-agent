from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..core.permissions import ToolPolicy


ToolHandler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    policy: ToolPolicy
    handler: ToolHandler

    def public_description(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "mutates_state": self.policy.mutates_state,
            "requires_confirmation": self.policy.requires_confirmation,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Duplicate tool: {definition.name}")
        self._tools[definition.name] = definition

    def describe(self) -> list[dict[str, Any]]:
        return [self._tools[name].public_description() for name in sorted(self._tools)]

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        required = set(schema.get("required", []))
        missing = sorted(required - set(arguments))
        if missing:
            raise ValueError(f"Missing tool arguments: {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(arguments) - set(properties))
            if extra:
                raise ValueError(f"Unknown tool arguments: {', '.join(extra)}")
        for name, value in arguments.items():
            rule = properties.get(name, {})
            expected = rule.get("type")
            if expected == "string":
                if not isinstance(value, str):
                    raise ValueError(f"Tool argument {name} must be a string.")
                if len(value) < int(rule.get("minLength", 0)):
                    raise ValueError(f"Tool argument {name} is too short.")
            elif expected == "array":
                if not isinstance(value, list):
                    raise ValueError(f"Tool argument {name} must be an array.")
                if len(value) < int(rule.get("minItems", 0)):
                    raise ValueError(f"Tool argument {name} has too few items.")
                item_type = rule.get("items", {}).get("type")
                if item_type == "string" and any(not isinstance(item, str) for item in value):
                    raise ValueError(f"Tool argument {name} must contain only strings.")
            elif expected == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Tool argument {name} must be a boolean.")
            if "enum" in rule and value not in rule["enum"]:
                raise ValueError(f"Tool argument {name} has an unsupported value.")

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        actor_type: str,
        actor_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc
        self._validate_arguments(tool.input_schema, arguments)
        tool.policy.authorize(actor_type, confirmed)
        context = {
            "actor_type": actor_type,
            "actor_id": actor_id,
            "confirmed": confirmed,
        }
        return tool.handler(arguments, context)


def build_phase1_registry(store: Any, model_storage: Any | None = None) -> ToolRegistry:
    if model_storage is None:
        from ..application.model_storage import ModelStorageService

        model_storage = ModelStorageService(store, store.path.parent.parent)

    def create_project(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        root = model_storage.validate_root(args["model_storage_root"])
        existed = root.exists()
        try:
            root.mkdir(parents=True, exist_ok=True)
            return store.create_project(
                project_name=args["project_name"],
                class_names=args["classes"],
                model_storage_root=str(root),
                actor_type=context["actor_type"],
                actor_id=context["actor_id"],
            )
        except Exception:
            if not existed:
                try:
                    root.rmdir()
                except OSError:
                    pass
            raise

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "delete_project",
            "Delete a local project after exact-name human confirmation.",
            {
                "type": "object",
                "required": ["project_id", "confirmation_name"],
                "properties": {
                    "project_id": {"type": "string", "minLength": 1},
                    "confirmation_name": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            ToolPolicy(
                mutates_state=True,
                requires_confirmation=True,
                allowed_actor_types=("human",),
            ),
            lambda args, ctx: store.delete_project(
                project_id=args["project_id"],
                confirmation_name=args["confirmation_name"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "create_project",
            "Create a local engineering project with stable class identifiers.",
            {
                "type": "object",
                "required": ["project_name", "classes", "model_storage_root"],
                "properties": {
                    "project_name": {"type": "string", "minLength": 1},
                    "model_storage_root": {"type": "string", "minLength": 1},
                    "classes": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "additionalProperties": False,
            },
            ToolPolicy(
                mutates_state=True,
                requires_confirmation=True,
                allowed_actor_types=("human",),
            ),
            create_project,
        )
    )
    registry.register(
        ToolDefinition(
            "confirm_taxonomy",
            "Freeze the engineer-confirmed image-level class taxonomy.",
            {
                "type": "object",
                "required": ["project_id"],
                "properties": {"project_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=True),
            lambda args, ctx: store.transition_project(
                project_id=args.get("project_id", ""),
                action="confirm_taxonomy",
                confirmed=ctx["confirmed"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "create_maintenance_batch",
            "Open one governed maintenance transaction for a Champion-ready project.",
            {
                "type": "object",
                "required": ["project_id", "batch_name"],
                "properties": {
                    "project_id": {"type": "string", "minLength": 1},
                    "batch_name": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=True),
            lambda args, ctx: store.create_maintenance_batch(
                project_id=args.get("project_id", ""),
                batch_name=args.get("batch_name", ""),
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "create_maintenance_dataset",
            "Create the single image-level dataset governed by an active maintenance batch.",
            {
                "type": "object",
                "required": ["batch_id", "name"],
                "properties": {
                    "batch_id": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=False),
            lambda args, ctx: store.create_maintenance_dataset(
                batch_id=args["batch_id"],
                name=args["name"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "freeze_maintenance_batch",
            "Freeze a validated maintenance dataset for Champion screening.",
            {
                "type": "object",
                "required": ["batch_id"],
                "properties": {"batch_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            ToolPolicy(
                mutates_state=True,
                requires_confirmation=True,
                allowed_actor_types=("human",),
            ),
            lambda args, ctx: store.freeze_maintenance_batch(
                batch_id=args["batch_id"],
                confirmed=ctx["confirmed"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "record_deployment_decision",
            "Record the engineer's single terminal Retain or Promote decision for an evaluated round.",
            {
                "type": "object",
                "required": ["batch_id", "decision", "reason"],
                "properties": {
                    "batch_id": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "enum": ["retain", "promote"]},
                    "reason": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            ToolPolicy(
                mutates_state=True,
                requires_confirmation=True,
                allowed_actor_types=("human",),
            ),
            lambda args, ctx: store.record_batch_decision(
                batch_id=args.get("batch_id", ""),
                decision=args.get("decision", ""),
                reason=args.get("reason", ""),
                confirmed=ctx["confirmed"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "import_dataset",
            "Create a governed local dataset after taxonomy confirmation.",
            {
                "type": "object",
                "required": ["project_id", "name", "role"],
                "properties": {
                    "project_id": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "enum": ["initial_training", "screening", "maintenance"]},
                },
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=False),
            lambda args, ctx: store.create_dataset(
                project_id=args["project_id"],
                name=args["name"],
                role=args["role"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "save_image_labels",
            "Save one engineer-provided image-level multi-label annotation.",
            {
                "type": "object",
                "required": ["image_id", "class_ids", "no_defect"],
                "properties": {
                    "image_id": {"type": "string", "minLength": 1},
                    "class_ids": {"type": "array"},
                    "no_defect": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=False),
            lambda args, ctx: store.save_annotation(
                image_id=args["image_id"],
                class_ids=args["class_ids"],
                no_defect=bool(args["no_defect"]),
            ),
        )
    )
    registry.register(
        ToolDefinition(
            "validate_dataset",
            "Run deterministic integrity and label checks, then freeze a valid label version.",
            {
                "type": "object",
                "required": ["dataset_id"],
                "properties": {"dataset_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            ToolPolicy(mutates_state=True, requires_confirmation=False),
            lambda args, ctx: store.validate_dataset(
                dataset_id=args["dataset_id"],
                actor_type=ctx["actor_type"],
                actor_id=ctx["actor_id"],
            ),
        )
    )
    return registry
