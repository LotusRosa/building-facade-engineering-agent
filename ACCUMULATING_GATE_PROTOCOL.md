# Accumulating Gate protocol

Train and evaluation data are separate in every round. Reserve an independent
initial Core Safety seed `S0` before initial training. Only initial Train images
enter Initial Champion training; import the reserved seed for the first paired
evaluation.

## Round contract

- Every maintenance round supplies its own Train/maintenance dataset `Dr`.
  Failure Discovery and expert review determine the update training data.
- Every evaluated round supplies a fresh, independent Current Gate `Gr`.
- The first evaluation also imports the reserved initial Core Safety `S0`.
- Later evaluations automatically attach cumulative Core Safety `S(r-1)`;
  they do not ask for another safety folder.
- Champion and Challenger are evaluated on both `Gr` and `S(r-1)` using
  their registered thresholds, without retuning or automatic selection.
- After successful evaluation and the human Retain or Promote decision,
  `Sr = S(r-1) union Gr`. The completed Gate is historical evaluation-only data.
  Either decision has the same Gate accumulation behavior.
- Failed, unfinished, or early-terminated rounds do not activate evaluation data.

Eligible Discovery datasets from completed rounds may enter later history replay.
Core Safety and Gates never enter training or replay. History is immutable,
with original cohort identities retained for provenance rather than duplicated
as extra evaluation splits.

## Human input

Use separate folders for each role. Every folder supports the same project
Excel/CSV template: `filename`, one 0/1 column per frozen class, and `no_defect`.
Missing labels can be completed inside the Agent. Required new evaluation
cohorts must be nonempty, fully annotated, and have positive support per class.
Suggested Train:holdout ratios are 7:3 or 8:2: initially the holdout is Core
Safety, and in each maintenance round it is Current Gate. These are optional,
not enforced or automatically applied. Keep related acquisitions together when
partitioning; exact-content checks cannot detect every near-duplicate image.

Evaluation folders may be imported and annotated early, but job freezing requires
a registered Challenger and explicit confirmation bound to the input revision.
Freezing does not launch a GPU job. The backend verifies checksums, taxonomy,
labels, class support, and zero overlap against Train and evaluation history.

## Internal snapshot manifest v4

The folder workflow generates this internal package; users do not construct ZIPs.
For the first evaluation its manifest includes:

```json
{
  "schema_version": 4,
  "purpose": "champion_challenger_selection",
  "project_id": "project_...",
  "batch_id": "batch_...",
  "provided_splits": ["current_gate", "core_safety"],
  "label_version_id": "folder-labels-...",
  "taxonomy_sha256": "..."
}
```

Later rounds use `"provided_splits": ["current_gate"]`. The assembled Worker
job always includes both logical evaluation splits, `current_gate` and
`core_safety`. It contains classes, label rows, referenced images, and checksums.
Legacy v3 seed packages remain compatible for the first evaluation; existing
historical cohort records remain immutable and readable.

## Terminal activation

New cohorts remain pending until the engineer records Retain or Promote.
The first completed round activates its safety seed and Gate. Later completed
rounds activate only their Gate and record the cumulative Core Safety count.
Retain keeps the Champion; Promote changes it only through the confirmed decision.
