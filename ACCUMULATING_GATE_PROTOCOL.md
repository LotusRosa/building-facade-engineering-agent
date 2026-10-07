# Accumulating Gate protocol

Every completed maintenance round contributes to two strictly separated histories:

- Its frozen Discovery dataset becomes eligible for later `history_replay`, whether the decision was Retain or Promote.
- Its current Gate becomes immutable evaluation-only history after the final Retain or Promote decision and is added to the cumulative Core Safety history.

Gate images and labels never enter training. The maintenance product has no terminal benchmark workflow.

## Round contract

At round `r`, the backend assembles:

- training history: initial/Core training data plus all earlier Discovery datasets;
- evaluation history: cumulative Core Safety, which contains the original safety cohort plus every earlier completed round Gate;
- current evaluation: exactly one newly supplied current Gate.

The engineer supplies only the current Gate after the first round. The backend attaches registered history, rejects missing or changed artifacts, and verifies image SHA-256 values against all project datasets and all registered Gates.

## Snapshot manifest v2

The first evaluation supplies `current_gate` and `core_safety`:

```json
{
  "schema_version": 2,
  "purpose": "deployment_decision_evidence",
  "project_id": "project_...",
  "batch_id": "batch_...",
  "current_gate_name": "round_a_gate",
  "provided_splits": ["current_gate", "core_safety"]
}
```

Every later evaluation supplies only `current_gate`:

```json
{
  "schema_version": 2,
  "purpose": "deployment_decision_evidence",
  "project_id": "project_...",
  "batch_id": "batch_...",
  "current_gate_name": "round_b_gate",
  "provided_splits": ["current_gate"]
}
```

`current_gate_name` must be a new lowercase identifier beginning with a letter and containing only letters, digits, and underscores. Each supplied Gate must have positive support for every frozen class.

The ZIP also contains `classes.json`, `labels.jsonl`, every referenced `images/...` member, and `checksums.json`. Each label row records `image_id`, `split`, `image_file`, `image_sha256`, `no_defect`, and `class_ids`.

## Terminal activation

The current Gate remains `pending` while evidence is reviewed. The single final Retain or Promote decision activates both the current Gate and its Core Safety addition in the round summary. Early termination before Challenger creation has no evaluation Gate and therefore activates none.
