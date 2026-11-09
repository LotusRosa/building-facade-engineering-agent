# Correspondence with the current manuscript

Associated manuscript: **Agent-Assisted Automation of Model Adaptation for UAV-Based Building Facade Inspection**. This note compares the current interactive software with the supplied manuscript source; it does not change frozen experiments or results.

## Corresponding components

| Manuscript component | Repository implementation / data |
|---|---|
| Four functional modules | Confirmed task preparation, model development, candidate evaluation and project state/version management |
| Three image-level defect outputs | `hollow`, `spalling`, `crack`; `no_defect` is the absence of all three |
| Initial and sequential data roles | 2000 initial Train; 400 Val; 400 Test0/S0; A and B each 300 Train + 100 Gate; 400 Final |
| Leakage-aware allocation | 4000 stable image IDs and 121 role-disjoint groups in `protocols/bfd_ml_4k/roles.csv` |
| Architectures | ResNet-50, ConvNeXt-Tiny, Swin-T; ImageNet initialization for initial modeling |
| Eight adaptation strategies | FT=`standard`, ASL=`asl`, SAM=`sam`, KD=`distill`, HNM=`hnm`, LWS=`hem`, Head-only=`head_only`, EWC=`ewc_replay` |
| Historical replay | All eight adaptation strategies use the common new/history schedule; default 8+8 draws, 38 steps per epoch, five epochs |
| Model selection and calibration | Initial best Val macro-AP; adaptation final scheduled state; Val F1 thresholds; paired parent/candidate evaluation |
| Engineer's decision | Confirmed adopt/Hold with a recorded reason; retained model and completed data pools carry to the next round |
| Architecture figure | Current manuscript Figure 1, with original dimensions/pixels preserved |

The frozen role file also matches all per-role positive counts in the manuscript's data-composition table. The initial pool is the full 2000-image allocation; preparing it does not generate the paper's six nested per-seed subsets.

## Confirmed difference: initial-training precision

The manuscript's common-training table specifies **FP16 with gradient scaling for initial construction**, BF16 for sequential adaptation, and FP16 evaluation. The current interactive Agent records and executes **BF16 training for both initial and adaptation jobs**, with FP16 evaluation (`facade_next/contracts.py` and `facade_next/gpu.py`).

This is an unresolved exact-reproduction difference for initial training. It must be reconciled with the original benchmark configuration and saved run records before claiming identical settings. The documentation audit did not switch precision, edit historical receipts, rerun experiments or modify the manuscript to conceal the difference. Existing initial/Standard Ubuntu workflow records remain evidence for those recorded jobs, not for every paper configuration.

## Scope of the public quickstart

The [first-install instructions](FIRST_INSTALL.md) run the interactive initial/A workflow. They do not automatically execute the paper's 54 trajectories, all three seeds, six nested initial sizes, cross-seed engineering selections, or the external CUBIT-Det experiment. The public fixed-role manifest establishes the internal corpus allocation; it is not the complete set of per-seed subset manifests and experiment outputs needed to reproduce all paper tables.

CUBIT-Det is a separate third-party corpus and is not included in BFD-ML-4K. Its target-only adaptation protocol, including the exclusion of source-image replay, is separate from the ordinary new/history update route. Do not present an ordinary five-epoch replay task as that external experiment.

The current manuscript source does not yet contain the canonical GitHub/Hugging Face availability links. Adding an author-approved Data/Code Availability paragraph is a manuscript action; repository documentation is not evidence that those links are already in the paper.

Windows/Ubuntu model-free checks and public download-route checks are separate from real GPU training validation. [Validation scope](VALIDATION.md) lists the recorded evidence and limits.
