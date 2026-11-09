# Prepare already-split data

[中文](DATA_PREPARATION_ZH.md)

The engineer supplies data with roles already assigned. Neither the Agent nor its optional LLM randomly divides, shuffles, or reallocates the submitted dataset. Provide paths on the Agent host; browser path submission is not an upload of all images.

## Initial modeling

Supply **Train**, independent **Val**, and initial acceptance **S0**. Train learns weights; Val selects the initial checkpoint and calibrates thresholds; S0 independently evaluates the resulting model. S0 never enters training, mining, Fisher estimation, or calibration. No additional research Final set is required.

```text
initial/
  train/images/             initial training photos
  train/labels.csv
  val/images/               independent validation photos
  val/labels.csv
  safety/images/            independent S0 photos
  safety/labels.csv
```

Create a project in chat and identify this batch folder. Review the discovered role mapping before confirming imports. Supported aliases also include `training`, `validation`, `s0`, `core_safety`, `test`, and `evaluation`, according to the role. Avoid multiple folders that ambiguously name the same role.

## Later adaptation rounds

Each round adds a new Train and a new independent Current Gate. New Val is optional; existing Val, training history, and historical acceptance are attached automatically.

```text
round_A/
  train/images/
  train/labels.csv
  gate/images/
  gate/labels.csv
```

Start another maintenance round in chat and submit its batch folder. Repeat for B, C, and subsequent batches. Names do not override the project's actual initial/adaptation stage. Do not submit a parent directory containing several batches as a single Train.

| Stage | Training | Validation/calibration | Independent evaluation |
|---|---|---|---|
| Initial | Train0 | Val0 | S0 |
| A | TrainA + permitted history | Existing Val; optional ValA | S0 + GateA |
| B | TrainB + permitted history | Accumulated Val; optional ValB | S0 + GateA + GateB |

After completed evaluation and an explicit adopt/Hold decision, a gate joins historical acceptance. Failed unevaluated or cancelled rounds do not accumulate it. Val and acceptance remain protected even when a round is cancelled; they cannot later be reassigned to training. Rollback is a separate operation.

## Label tables

Use UTF-8 CSV or the first worksheet of XLSX. Tables contain values, not formulas. The current defect classes are `hollow`, `spalling`, and `crack`.

```csv
image_id,path,hollow,spalling,crack,scene,group
photo_001,building_a/001.jpg,,0,1,scene_a,building_a
```

`path` is relative to the declared image root, or an absolute path inside it. The legacy `filename,hollow,spalling,crack,no_defect` layout is supported; examples are in [templates](../templates). A `labels.csv` or `labels.xlsx` next to `images/` is detected automatically. Multiple matching tables require explicit selection.

- `1` means present; `0` means confirmed absent; blank/unknown remains unknown.
- Multiple defect positives are allowed. `no_defect=1` requires explicit zeros for every defect column and is not an extra model category.
- Partial labels are supported. Each training photo needs at least one known label. Missing labels are not silently made negative.
- Initial three-class Train and Val must each provide positive and negative support for every class. For supported adaptation with incomplete external labels, unknown outputs stay masked and uncalibratable thresholds can inherit the parent's values.
- An explicit `image_id` must remain stable. Without one, a content-derived identity is generated. Duplicate IDs/content within a subset and incompatible cross-role overlap are rejected.
- `scene` identifies evaluation scenes. `group` identifies acquisition/building groups. Keep related images in one role; exact hashes cannot identify every near-duplicate or related acquisition if groups are absent.

Complete labels bypass repeat annotation; partial/unannotated photos can continue through the existing annotation controls. The task freezes actual manifests and photo identities. Keep source files and paths stable; the Worker rechecks image hashes.

Use independent runtime directories across hosts. Do not move a Windows task database with frozen Windows paths into Ubuntu and assume its files have been remapped. Shared environment setup is separate from dataset and runtime-directory selection.
